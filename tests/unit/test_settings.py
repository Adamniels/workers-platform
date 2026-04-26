"""Unit tests for runtime settings parsing."""

from app.runtime.config.settings import Settings


def test_settings_read_environment_aliases(monkeypatch) -> None:
    """Settings should load expected environment variables."""
    monkeypatch.setenv("TEMPORAL_SERVER_URL", "127.0.0.1:7233")
    monkeypatch.setenv("TEMPORAL_NAMESPACE", "platform-dev")
    monkeypatch.setenv("TEMPORAL_TASK_QUEUE", "platform")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.temporal_server_url == "127.0.0.1:7233"
    assert settings.temporal_namespace == "platform-dev"
    assert settings.temporal_task_queue == "platform"
    assert settings.log_level == "DEBUG"
