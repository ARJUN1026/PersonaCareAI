from src.config import config
from src.rag_pipeline import LocalRAGPipeline

if __name__ == "__main__":
    if not config.has_gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required. Copy .env.example to .env and add your actual Gemini key."
        )

    pipeline = LocalRAGPipeline()
    count = pipeline.ingest_all(reset=True)
    print(f"✅ Ingested {count} chunks into ChromaDB")
    print(f"📦 Collection now contains {pipeline.count()} chunks")
