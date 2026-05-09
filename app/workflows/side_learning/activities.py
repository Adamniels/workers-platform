"""Activities for side-learning workflows (Stages A, B, and C)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from temporalio import activity

from app.memory.client.http_client import PlatformMemoryHttpClient
from app.memory.client.models import GetMemoryContextV1Request, MemoryContextV1
from app.runtime.config import Settings, get_settings
from app.workflows.side_learning.contracts import (
    SessionContentLlmResponse,
    TopicProposalItem,
    TopicProposalLlmResponse,
    TopicSelectionMemoryProposalsLlmResponse,
)
from app.workflows.side_learning.memory_mapper import (
    build_topic_proposal_system_prompt,
    build_topic_proposal_user_prompt,
    filter_proposals_against_recalls,
)
from app.workflows.side_learning.session_stage_b import (
    build_context_section_system_prompt,
    build_context_section_user_prompt,
    build_session_generation_system_prompt,
    build_session_generation_user_prompt,
    build_topic_memory_system_prompt,
    build_topic_memory_user_prompt,
    normalize_session_sections,
    wire_memory_proposals_from_llm,
)
from app.workflows.side_learning.session_stage_c import (
    build_reflection_memory_system_prompt,
    build_reflection_memory_user_prompt,
)

logger = logging.getLogger(__name__)


def _side_learning_session_llm_model(settings: Settings) -> str:
    alt = (settings.openai_side_learning_session_model or "").strip()
    return alt or settings.openai_model


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


@activity.defn
async def fetch_memory_context_for_session_generation(
    topic_title: str,
    user_feedback: str | None = None,
) -> dict[str, Any]:
    """Load memory context; taskDescription = topic (feedback used in later activities)."""
    _ = user_feedback
    settings = get_settings()
    client = PlatformMemoryHttpClient(settings)
    td = (topic_title or "").strip() or None
    body = GetMemoryContextV1Request(
        user_id=settings.consolidation_primary_user_id,
        task_description=td,
        workflow_type="side_learning",
        domain="learning",
        include_vector_recall=True,
    )
    ctx = await client.post_memory_context(body)
    return ctx.model_dump(mode="json", by_alias=True)


@activity.defn
async def generate_learning_session(
    context_dict: dict[str, Any],
    topic_title: str,
    user_feedback: str | None,
) -> list[dict[str, Any]]:
    """LLM: four fixed sections for the learning session."""
    settings = get_settings()
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY is not set; cannot generate session."
        logger.error(msg)
        raise RuntimeError(msg)

    context = MemoryContextV1.model_validate(context_dict)
    system = build_session_generation_system_prompt()
    user = build_session_generation_user_prompt(context, topic_title, user_feedback)
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    model = _side_learning_session_llm_model(settings)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.65,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "OpenAI session generation failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = SessionContentLlmResponse.model_validate(json.loads(content))
    normalized = normalize_session_sections(parsed.sections)
    logger.debug(
        "generate_learning_session model=%s section_ids=%s",
        model,
        [s.get("id") for s in normalized],
    )
    return normalized


@activity.defn
async def generate_context_section(
    context_dict: dict[str, Any],
    topic_title: str,
    user_feedback: str | None,
) -> str:
    """LLM: write the full teaching content for the context section (plain Markdown)."""
    settings = get_settings()
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY is not set; cannot generate context section."
        logger.error(msg)
        raise RuntimeError(msg)

    context = MemoryContextV1.model_validate(context_dict)
    system = build_context_section_system_prompt()
    user = build_context_section_user_prompt(context, topic_title, user_feedback)
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    model = _side_learning_session_llm_model(settings)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "OpenAI context section generation failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    text = (content or "").strip()
    logger.debug("generate_context_section model=%s chars=%s", model, len(text))
    return text


@activity.defn
async def analyze_topic_selection_for_memory(
    context_dict: dict[str, Any],
    topic_title: str,
    user_feedback: str | None,
) -> list[dict[str, Any]]:
    """LLM: memory review proposals (MVP: NewSemantic, NewProceduralRule only)."""
    settings = get_settings()
    if not settings.openai_api_key:
        return []

    context = MemoryContextV1.model_validate(context_dict)
    system = build_topic_memory_system_prompt()
    user = build_topic_memory_user_prompt(context, topic_title, user_feedback)
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    model = _side_learning_session_llm_model(settings)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "OpenAI topic memory analysis failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = TopicSelectionMemoryProposalsLlmResponse.model_validate(json.loads(content))
    return wire_memory_proposals_from_llm(parsed.proposals, max_proposals=3)


@activity.defn
async def fetch_memory_context_for_reflection(
    topic_title: str,
    reflection_text: str,
) -> dict[str, Any]:
    """Load memory context; taskDescription = topic + reflection excerpt for retrieval."""
    settings = get_settings()
    client = PlatformMemoryHttpClient(settings)
    topic = (topic_title or "").strip()
    ref = (reflection_text or "").strip()
    excerpt = ref[:1200] if ref else ""
    parts = [topic] if topic else []
    if excerpt:
        parts.append(f"User reflection (excerpt):\n{excerpt}")
    td = "\n\n".join(parts).strip() or None
    body = GetMemoryContextV1Request(
        user_id=settings.consolidation_primary_user_id,
        task_description=td,
        workflow_type="side_learning",
        domain="learning",
        include_vector_recall=True,
    )
    ctx = await client.post_memory_context(body)
    return ctx.model_dump(mode="json", by_alias=True)


@activity.defn
async def analyze_reflection(
    context_dict: dict[str, Any],
    topic_title: str,
    reflection_text: str,
    session_content_json: str,
) -> list[dict[str, Any]]:
    """LLM: memory proposals from reflection + session (MVP types)."""
    settings = get_settings()
    if not settings.openai_api_key:
        return []

    context = MemoryContextV1.model_validate(context_dict)
    system = build_reflection_memory_system_prompt()
    user = build_reflection_memory_user_prompt(
        context,
        topic_title,
        reflection_text,
        session_content_json,
    )
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    model = _side_learning_session_llm_model(settings)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "OpenAI reflection analysis failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = TopicSelectionMemoryProposalsLlmResponse.model_validate(json.loads(content))
    # Uses session_stage_b.wire_memory_proposals_from_llm (internally try_wire_memory_proposal).
    return wire_memory_proposals_from_llm(parsed.proposals, max_proposals=5)


@activity.defn
async def post_reflection_insights(
    session_id: str,
    memory_proposals: list[dict[str, Any]],
) -> None:
    """POST /api/internal/v1/side-learning/sessions/{id}/reflection-insights"""
    settings = get_settings()
    if not settings.platform_internal_service_token:
        raise RuntimeError(
            "PLATFORM_INTERNAL_SERVICE_TOKEN (or legacy MEMORY_WORKER_SERVICE_TOKEN) is not set."
        )
    base = settings.platform_api_base_url.rstrip("/")
    url = f"{base}/api/internal/v1/side-learning/sessions/{session_id}/reflection-insights"
    headers = {"Authorization": f"Bearer {settings.platform_internal_service_token}"}
    body: dict[str, Any] = {"memoryProposals": memory_proposals or []}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "post_reflection_insights failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise


@activity.defn
async def post_session_content(
    session_id: str,
    sections: list[dict[str, Any]],
    memory_proposals: list[dict[str, Any]],
) -> None:
    """POST /api/internal/v1/side-learning/sessions/{id}/session-content"""
    settings = get_settings()
    if not settings.platform_internal_service_token:
        raise RuntimeError(
            "PLATFORM_INTERNAL_SERVICE_TOKEN (or legacy MEMORY_WORKER_SERVICE_TOKEN) is not set."
        )
    base = settings.platform_api_base_url.rstrip("/")
    url = f"{base}/api/internal/v1/side-learning/sessions/{session_id}/session-content"
    headers = {"Authorization": f"Bearer {settings.platform_internal_service_token}"}
    body: dict[str, Any] = {"sections": sections}
    if memory_proposals:
        body["memoryProposals"] = memory_proposals
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=body, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception(
            "post_session_content failed status=%s body=%s",
            response.status_code,
            response.text[:2000],
        )
        raise
