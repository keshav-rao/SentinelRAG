import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env into os.environ for LangSmith tracing
load_dotenv(BASE_DIR / ".env", override=True)

class Settings(BaseSettings):
    # LangSmith Observability
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="SentinelRAG", alias="LANGSMITH_PROJECT")

    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field(default="qwen3:4b-instruct", alias="OLLAMA_LLM_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    ollama_api_key: str = Field(default="", alias="OLLAMA_API_KEY")

    # Qdrant Settings
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="sentinel_knowledge", alias="QDRANT_COLLECTION_NAME")
    qdrant_local_path: str = Field(default="./qdrant_storage", alias="QDRANT_LOCAL_PATH")

    # RAG & Guardrail Settings
    top_k_retrieval: int = Field(default=6, alias="TOP_K_RETRIEVAL")
    top_k_reranked: int = Field(default=2, alias="TOP_K_RERANKED")
    critic_min_score: float = Field(default=0.80, alias="CRITIC_MIN_SCORE")
    max_retrieval_retries: int = Field(default=2, alias="MAX_RETRIEVAL_RETRIES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Directory Paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    eval_data_dir: Path = BASE_DIR / "data" / "evaluation"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
