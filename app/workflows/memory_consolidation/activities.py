"""Activities: call internal .NET consolidation API."""

import logging

import httpx
from temporalio import activity

from app.runtime.config import get_settings

logger = logging.getLogger(__name__)


@activity.defn
async def run_nightly_memory_consolidation_activity() -> dict:
    """POST nightly consolidation; bearer auth; deterministic work runs in .NET."""
    settings = get_settings()
    if not settings.memory_worker_token:
        msg = "MEMORY_WORKER_SERVICE_TOKEN is not set; cannot call internal consolidation API."
        logger.error(msg)
        raise RuntimeError(msg)

    base = settings.platform_api_base_url.rstrip("/")
    url = f"{base}/api/internal/v1/memory/consolidation/nightly"
    body = {"userId": settings.consolidation_primary_user_id}
    headers = {"Authorization": f"Bearer {settings.memory_worker_token}"}

    logger.info("calling consolidation API %s userId=%s", url, body["userId"])
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "consolidation API failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    logger.info(
        "consolidation done runId=%s fromCache=%s proposals=%s auto=%s",
        data.get("runId"),
        data.get("fromCache"),
        data.get("proposalsCreatedCount"),
        data.get("autoUpdatesCount"),
    )
    return data
