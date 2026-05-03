"""HTTP client for Platform .NET internal memory APIs."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.memory.client.models import GetMemoryContextV1Request, MemoryContextV1
from app.runtime.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PlatformMemoryHttpClient:
    """Calls internal worker endpoints on the Platform API (Bearer auth)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _headers(self) -> dict[str, str]:
        token = self._settings.platform_internal_service_token
        if not token:
            raise RuntimeError(
                "PLATFORM_INTERNAL_SERVICE_TOKEN (or legacy MEMORY_WORKER_SERVICE_TOKEN) is not set; "
                "must match PlatformWorkers:ServiceToken on the API.",
            )
        return {"Authorization": f"Bearer {token}"}

    def _base(self) -> str:
        return self._settings.platform_api_base_url.rstrip("/")

    async def post_memory_context(self, body: GetMemoryContextV1Request) -> MemoryContextV1:
        """POST /api/internal/v1/memory/context — same contract as public v1."""
        url = f"{self._base()}/api/internal/v1/memory/context"
        payload: dict[str, Any] = body.model_dump(mode="json", by_alias=True, exclude_none=True)
        logger.debug("POST memory context %s keys=%s", url, list(payload.keys()))
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=self._headers())
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.exception(
                "memory context API failed status=%s body=%s",
                response.status_code,
                response.text[:2000],
            )
            raise
        return MemoryContextV1.model_validate(response.json())
