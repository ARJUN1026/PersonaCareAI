from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import config
from .utils import call_with_backoff


class LocalRAGPipeline:
    """Gemini embeddings + ChromaDB persistent vector store for support documentation."""

    def __init__(self, db_dir: str | Path | None = None):
        self.client = genai.Client(api_key=config.gemini_api_key) if config.has_gemini_api_key else None
        self.chroma_client = chromadb.PersistentClient(path=str(db_dir or config.chroma_db_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_embedding(self, text: str) -> list[float]:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings. Add it to .env.")
        try:
            response = call_with_backoff(
                self.client.models.embed_content,
                model=config.embedding_model,
                contents=text,
            )
            return response.embeddings[0].values
        except Exception as exc:
            message = str(exc)
            if "text-embedding-004" in message or "NOT_FOUND" in message or "models/" in message:
                raise RuntimeError(
                    f"Embedding model '{config.embedding_model}' is not available for this Gemini API version. "
                    "Use a supported model such as gemini-embedding-2, gemini-embedding-2-preview, or gemini-embedding-001. "
                    "Update GEMINI_EMBEDDING_MODEL in .env and restart."
                ) from exc
            raise

    @staticmethod
    def _read_txt_md(path: Path) -> list[dict[str, Any]]:
        return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "source": path.name, "page": None}]

    @staticmethod
    def _read_pdf(path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(path))
        pages: list[dict[str, Any]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            pages.append({"text": page.extract_text() or "", "source": path.name, "page": page_num})
        return pages

    def load_documents(self, data_dir: str | Path | None = None) -> list[dict[str, Any]]:
        data_path = Path(data_dir or config.data_dir)
        docs: list[dict[str, Any]] = []
        for path in sorted(data_path.glob("**/*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".txt", ".md"}:
                docs.extend(self._read_txt_md(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self._read_pdf(path))
        return docs

    def ingest_all(self, data_dir: str | Path | None = None, reset: bool = True) -> int:
        if reset:
            try:
                self.chroma_client.delete_collection(config.collection_name)
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_docs = self.load_documents(data_dir)
        inserted = 0
        for doc in raw_docs:
            chunks = splitter.split_text(doc["text"])
            for idx, chunk in enumerate(chunks):
                clean_chunk = chunk.strip()
                if len(clean_chunk) < 50:
                    continue
                source = doc["source"]
                page = doc.get("page")
                chunk_id_raw = f"{source}:{page}:{idx}:{clean_chunk[:60]}"
                chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()[:24]
                embedding = self.get_embedding(clean_chunk)
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{"source": source, "page": page or "section", "chunk_index": idx}],
                    documents=[clean_chunk],
                )
                inserted += 1
        return inserted

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        query_vector = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k or config.top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved_items: list[dict[str, Any]] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        dists = results.get("distances") or [[]]
        for i, text in enumerate(docs[0]):
            distance = dists[0][i] if dists and dists[0] else 1.0
            retrieved_items.append({
                "text": text,
                "source": metas[0][i].get("source", "unknown"),
                "page": metas[0][i].get("page", "section"),
                "chunk_index": metas[0][i].get("chunk_index", i),
                "score": max(0.0, min(1.0, 1.0 - float(distance))),
            })
        return retrieved_items

    def count(self) -> int:
        return self.collection.count()
