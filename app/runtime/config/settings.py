"""Typed runtime settings loaded from environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, Field
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
    platform_internal_service_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "PLATFORM_INTERNAL_SERVICE_TOKEN",
            "MEMORY_WORKER_SERVICE_TOKEN",
        ),
        description=(
            "Shared Bearer token for every worker calling Platform /api/internal/v1/*; "
            "must match PlatformWorkers:ServiceToken on the API host."
        ),
    )
    consolidation_primary_user_id: int = Field(default=1, alias="CONSOLIDATION_PRIMARY_USER_ID")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_side_learning_session_model: str = Field(
        default="",
        alias="OPENAI_SIDE_LEARNING_SESSION_MODEL",
        description=(
            "Optional override for side-learning Stage B LLM; "
            "falls back to OPENAI_MODEL when empty."
        ),
    )

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
