"""Unit tests for side-learning Stage C (reflection analysis)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.memory.client.models import MemoryContextV1
from app.runtime.config import Settings
from app.workflows.side_learning import activities
from app.workflows.side_learning.contracts import (
    TopicSelectionMemoryProposalLlmItem,
    TopicSelectionMemoryProposalsLlmResponse,
)
from app.workflows.side_learning.session_stage_c import (
    build_reflection_memory_system_prompt,
    build_reflection_memory_user_prompt,
)


def test_reflection_prompts_include_topic_and_truncation() -> None:
    ctx = MemoryContextV1()
    long_json = '{"sections":[' + ("x" * 20_000) + ']}'
    user = build_reflection_memory_user_prompt(ctx, "Topic T", "My reflection", long_json)
    assert "Topic T" in user
    assert "My reflection" in user
    assert "truncated" in user.lower() or len(user) < len(long_json) + 500
    assert len(build_reflection_memory_system_prompt()) > 40


def _async_client_with_transport(transport: httpx.MockTransport):
    real_ctor = httpx.AsyncClient

    def factory(*_a: object, **kw: object) -> httpx.AsyncClient:
        return real_ctor(transport=transport, **kw)

    return factory


@pytest.mark.asyncio
async def test_post_reflection_insights_posts_memory_proposals() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    settings = MagicMock(spec=Settings)
    settings.platform_internal_service_token = "tok"
    settings.platform_api_base_url = "http://test"
    patch_client = patch.object(
        activities.httpx,
        "AsyncClient",
        side_effect=_async_client_with_transport(transport),
    )
    mem = [
        {
            "proposalType": "NewSemantic",
            "title": "t",
            "summary": "",
            "proposedChangeJson": "{}",
            "priority": 0,
        },
    ]
    with patch.object(activities, "get_settings", return_value=settings), patch_client:
        await activities.post_reflection_insights("sess-1", mem)
    assert captured["body"]["memoryProposals"][0]["proposalType"] == "NewSemantic"


@pytest.mark.asyncio
async def test_analyze_reflection_uses_max_five_wires() -> None:
    proposals = []
    for i in range(6):
        proposals.append(
            TopicSelectionMemoryProposalLlmItem(
                proposal_type="NewSemantic",
                title=f"t{i}",
                summary="s",
                proposed_change={
                    "kind": "NewSemantic",
                    "key": f"k{i}",
                    "claim": f"c{i}",
                    "domain": "learning",
                    "initialConfidence": 0.5,
                },
            ),
        )
    payload = TopicSelectionMemoryProposalsLlmResponse(proposals=proposals).model_dump(
        mode="json",
        by_alias=True,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    transport = httpx.MockTransport(handler)
    settings = MagicMock(spec=Settings)
    settings.openai_api_key = "k"
    settings.openai_base_url = "http://openai/v1"
    settings.openai_model = "m"
    settings.openai_side_learning_session_model = ""
    ctx = MemoryContextV1().model_dump(mode="json", by_alias=True)
    patch_client = patch.object(
        activities.httpx,
        "AsyncClient",
        side_effect=_async_client_with_transport(transport),
    )
    with patch.object(activities, "get_settings", return_value=settings), patch_client:
        out = await activities.analyze_reflection(
            ctx,
            "Topic",
            "reflection text",
            '{"sections":[]}',
        )
    assert len(out) == 5


@pytest.mark.asyncio
async def test_fetch_memory_context_for_reflection_calls_client() -> None:
    ctx = MemoryContextV1(profile_facts=[])
    mock_client = MagicMock()
    mock_client.post_memory_context = AsyncMock(return_value=ctx)
    with patch.object(activities, "get_settings") as gs:
        gs.return_value.consolidation_primary_user_id = 1
        with patch.object(activities, "PlatformMemoryHttpClient", return_value=mock_client):
            out = await activities.fetch_memory_context_for_reflection(
                "Go modules",
                "I liked the exercises.",
            )
    mock_client.post_memory_context.assert_awaited_once()
    body = mock_client.post_memory_context.await_args.args[0]
    assert "Go modules" in (body.task_description or "")
    assert "reflection" in (body.task_description or "").lower()
    assert out["profileFacts"] == []
