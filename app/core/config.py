from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    app_name : str = "Agentic Bot Builder"
    app_env :str = Field(default= "development", env = "APP_ENV")
    app_host : str = Field(default= "0.0.0.0", env = "APP_HOST")
    app_port : int = Field(default= 8000, env = "APP_PORT")
    debug: bool = Field(default= True, env = "DEBUG")

    anthropic_api_key : str = Field(..., env = "ANTHROPIC_API_KEY")
    claude_model : str = Field(default= "claude-3-5-sonnet-20241022", env = "CLAUDE_MODEL")

    chroma_persist_dir : str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_prefix : str = Field(default="bot_", env="CHROMA_COLLECTION_PREFIX")

    embedding_model : str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")

    chunk_size : int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap : int = Field(default=64, env="CHUNK_OVERLAP")

    top_k_results : int = Field(default=5, env="TOP_K_RESULTS")
    similarity_threshold : float = Field(default=0.35, env="SIMILARITY_THRESHOLD")

    database_url: str = Field(default="sqlite+aiosqlite:///./data/agentic_bots.db", env="DATABASE_URL",)

    max_upload_size_mb : int = Field(default=50, env = "MAX_UPLOAD_SIZE_MB")
    upload_dir : str = Field(default= "./data/uploads", env = "UPLOAD_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def max_uplaod_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

@lru_cache()
def get_settings() -> Settings:
    return Settings()