"""Integration placeholder test for worker bootstrap wiring."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.runtime.worker import main as worker_main


@pytest.mark.asyncio
async def test_run_worker_builds_worker_with_registry(monkeypatch) -> None:
    """Worker run path should connect client and construct a Worker."""
    fake_settings = SimpleNamespace(
        log_level="INFO",
        temporal_task_queue="platform",
        temporal_server_url="localhost:7233",
        temporal_namespace="default",
    )

    fake_definitions = SimpleNamespace(workflows=[object], activities=[lambda: None])
    fake_client = object()

    worker_ctor = Mock(return_value=SimpleNamespace(run=AsyncMock()))

    monkeypatch.setattr(worker_main, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(worker_main, "configure_logging", lambda _: None)
    monkeypatch.setattr(worker_main, "get_registered_definitions", lambda: fake_definitions)
    monkeypatch.setattr(worker_main, "get_temporal_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr(worker_main, "Worker", worker_ctor)

    await worker_main.run_worker()

    worker_ctor.assert_called_once()
