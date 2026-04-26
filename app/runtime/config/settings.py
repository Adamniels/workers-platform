"""Typed runtime settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for Temporal worker execution."""

    temporal_server_url: str = Field(default="localhost:7233", alias="TEMPORAL_SERVER_URL")
    temporal_namespace: str = Field(default="default", alias="TEMPORAL_NAMESPACE")
    temporal_task_queue: str = Field(default="platform", alias="TEMPORAL_TASK_QUEUE")
    temporal_task_queue_memory: str = Field(
        default="memory-consolidation", alias="TEMPORAL_TASK_QUEUE_MEMORY"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    platform_api_base_url: str = Field(
        default="http://localhost:5120", alias="PLATFORM_API_BASE_URL"
    )
    memory_worker_token: str = Field(default="", alias="MEMORY_WORKER_SERVICE_TOKEN")
    consolidation_primary_user_id: int = Field(default=1, alias="CONSOLIDATION_PRIMARY_USER_ID")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-cached runtime settings."""
    return Settings()
