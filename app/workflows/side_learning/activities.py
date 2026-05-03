"""Activities for side-learning workflows (Stage A: topic proposals)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from temporalio import activity

from app.memory.client.http_client import PlatformMemoryHttpClient
from app.memory.client.models import GetMemoryContextV1Request, MemoryContextV1
from app.runtime.config import get_settings
from app.workflows.side_learning.contracts import TopicProposalItem, TopicProposalLlmResponse
from app.workflows.side_learning.memory_mapper import (
    build_topic_proposal_system_prompt,
    build_topic_proposal_user_prompt,
    filter_proposals_against_recalls,
)

logger = logging.getLogger(__name__)


@activity.defn
async def fetch_memory_context_for_learning(initial_prompt: str | None) -> dict[str, Any]:
    """Load MemoryContextV1 from internal .NET API."""
    settings = get_settings()
    client = PlatformMemoryHttpClient(settings)
    body = GetMemoryContextV1Request(
        user_id=settings.consolidation_primary_user_id,
        task_description=(initial_prompt or "").strip() or None,
        workflow_type="side_learning",
        domain="learning",
        include_vector_recall=True,
    )
    ctx = await client.post_memory_context(body)
    return ctx.model_dump(mode="json", by_alias=True)


@activity.defn
async def propose_learning_topics(
    context_dict: dict[str, Any],
    initial_prompt: str | None,
) -> list[dict[str, Any]]:
    """LLM: propose exactly five learning topics as structured JSON."""
    settings = get_settings()
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY is not set; cannot propose topics."
        logger.error(msg)
        raise RuntimeError(msg)

    context = MemoryContextV1.model_validate(context_dict)
    system = build_topic_proposal_system_prompt()
    user = build_topic_proposal_user_prompt(context, initial_prompt)
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "OpenAI chat failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = TopicProposalLlmResponse.model_validate(json.loads(content))
    topics = parsed.topics[:5]
    if len(topics) < 5:
        logger.warning("LLM returned fewer than 5 topics: count=%s", len(topics))
    return [t.model_dump(mode="json", by_alias=True) for t in topics]


@activity.defn
async def filter_known_topics(
    proposals: list[dict[str, Any]],
    context_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Remove topics too similar to document vector recalls; keep at least 3 when possible."""
    context = MemoryContextV1.model_validate(context_dict)
    items = [TopicProposalItem.model_validate(p) for p in proposals]
    filtered = filter_proposals_against_recalls(items, context)
    return [t.model_dump(mode="json", by_alias=True) for t in filtered]


@activity.defn
async def post_topic_proposals(session_id: str, topics: list[dict[str, Any]]) -> None:
    """POST /api/internal/v1/side-learning/sessions/{id}/proposals"""
    settings = get_settings()
    if not settings.platform_internal_service_token:
        raise RuntimeError(
            "PLATFORM_INTERNAL_SERVICE_TOKEN (or legacy MEMORY_WORKER_SERVICE_TOKEN) is not set."
        )
    base = settings.platform_api_base_url.rstrip("/")
    url = f"{base}/api/internal/v1/side-learning/sessions/{session_id}/proposals"
    headers = {"Authorization": f"Bearer {settings.platform_internal_service_token}"}
    body = {"topics": topics}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "post_topic_proposals failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
