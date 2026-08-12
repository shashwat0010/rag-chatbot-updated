from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Always load backend/.env regardless of shell cwd when starting uvicorn
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mistral AI / OpenRouter settings
    mistral_api_key: str = ""
    mistral_model: str = "openrouter/auto"
    mistral_concurrency_limit: int = 1
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    pubmed_max_results: int = 15
    pubmed_retrieval_top_k: int = 8
    min_relevance_score: float = 0.35
    min_evidence_chunks: int = 3

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    rate_limit_per_minute: int = 20
    log_level: str = "INFO"

    block_emergency_keywords: bool = True
    min_confidence_threshold: float = 0.4

    # WhatsApp integration settings
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""

    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""

    # Pinecone Vector DB settings
    pinecone_api_key: str = ""
    pinecone_index_name: str = "medical-rag-index"

    # Elasticsearch settings
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str = ""
    elasticsearch_index: str = "medical_papers_hybrid"

    # Authentication & Persistence settings
    jwt_secret_key: str = "supersecretjwtkey_change_in_production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    database_url: str = "sqlite:///./medical_rag.db"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
