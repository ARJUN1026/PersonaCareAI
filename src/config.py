from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
possible_env_files = [
    Path.cwd() / ".env",
    project_root / ".env",
    project_root / "src" / ".env",
]
for env_path in possible_env_files:
    if env_path.exists():
        load_dotenv(env_path, override=True)

@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    chroma_db_dir: Path = Path(os.getenv("CHROMA_DB_DIR", "./chroma_db"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    collection_name: str = os.getenv("COLLECTION_NAME", "support_kb")
    top_k: int = int(os.getenv("TOP_K", "3"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    escalate_after_dissatisfied: int = int(os.getenv("ESCALATE_AFTER_DISSATISFIED", "2"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    @property
    def has_gemini_api_key(self) -> bool:
        key = self.gemini_api_key.strip()
        if not key:
            return False
        placeholder_indicators = ("your_actual", "your_", "replace_me", "dummy")
        return not any(indicator in key.lower() for indicator in placeholder_indicators)

    @property
    def sensitive_keywords(self) -> tuple[str, ...]:
        return (
            "billing", "invoice", "refund", "duplicate charge", "payment", "credit card",
            "legal", "contract", "account ownership", "delete account", "data export",
            "security breach", "hacked", "unauthorized", "compliance"
        )

    @property
    def negative_keywords(self) -> tuple[str, ...]:
        return (
            "nothing works", "still not working", "angry", "frustrated", "useless",
            "terrible", "tried everything", "human", "manager", "escalate", "cancel"
        )

config = AppConfig()
