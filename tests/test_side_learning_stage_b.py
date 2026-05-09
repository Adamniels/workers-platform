"""Unit tests for side-learning Stage B (session content + memory proposals)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.memory.client.models import MemoryContextV1
from app.runtime.config import Settings
from app.workflows.side_learning import activities
from app.workflows.side_learning.contracts import (
    SideLearningSessionSection,
    TopicSelectionMemoryProposalLlmItem,
    TopicSelectionMemoryProposalsLlmResponse,
)
from app.workflows.side_learning.session_stage_b import (
    build_context_section_user_prompt,
    normalize_session_sections,
    wire_memory_proposals_from_llm,
)


def test_normalize_session_sections_fills_four_ids() -> None:
    raw = [
        SideLearningSessionSection(
            id="goal",
            label="Goal",
            estimated_minutes=5,
            type="goal",
            content="Learn X",
            example="Build a tiny demo",
        ),
        SideLearningSessionSection(
            id="context",
            label="Context",
            estimated_minutes=10,
            type="context",
            content="Why X matters",
            youtube_query="X tutorial beginner",
        ),
        SideLearningSessionSection(
            id="hands-on",
            label="Hands-on",
            estimated_minutes=40,
            type="hands-on",
            content="Do the exercise",
            output_type="code",
        ),
        SideLearningSessionSection(
            id="reflection",
            label="Reflection",
            estimated_minutes=10,
            type="reflection",
            content="",
            prompts=["Q1", "Q2", "Q3"],
        ),
    ]
    out = normalize_session_sections(raw)
    assert [s["id"] for s in out] == ["goal", "context", "hands-on", "reflection"]
    assert out[0]["example"] == "Build a tiny demo"
    assert out[1]["youtubeQuery"] == "X tutorial beginner"
    assert out[2]["outputType"] == "code"
    assert len(out[3]["prompts"]) == 3


def test_wire_memory_proposals_filters_invalid() -> None:
    items = [
        TopicSelectionMemoryProposalLlmItem(
            proposal_type="NewSemantic",
            title="t1",
            summary="s",
            proposed_change={
                "kind": "NewSemantic",
                "key": "k1",
                "claim": "c1",
                "domain": "learning",
                "initialConfidence": 0.7,
            },
        ),
        TopicSelectionMemoryProposalLlmItem(
            proposal_type="BadType",
            title="x",
            proposed_change={},
        ),
    ]
    out = wire_memory_proposals_from_llm(items)
    assert len(out) == 1
    assert out[0]["proposalType"] == "NewSemantic"
    assert "key" in out[0]["proposedChangeJson"]


def test_build_context_section_user_prompt_inserts_user_focus() -> None:
    ctx = MemoryContextV1()
    text = build_context_section_user_prompt(ctx, "Algebra", "more exercises")
    lines = text.split("\n")
    assert lines[0] == "Topic: Algebra"
    assert lines[1] == "User focus: more exercises"


def test_side_learning_session_llm_model_fallback() -> None:
    s = Settings(
        OPENAI_MODEL="base-model",
        OPENAI_SIDE_LEARNING_SESSION_MODEL="",
    )
    assert activities._side_learning_session_llm_model(s) == "base-model"
    s2 = Settings(
        OPENAI_MODEL="base-model",
        OPENAI_SIDE_LEARNING_SESSION_MODEL="session-model",
    )
    assert activities._side_learning_session_llm_model(s2) == "session-model"


def _async_client_with_transport(transport: httpx.MockTransport):
    """Bind real AsyncClient ctor before patch to avoid recursion via mocked name."""

    real_ctor = httpx.AsyncClient

    def factory(*_a: object, **kw: object) -> httpx.AsyncClient:
        return real_ctor(transport=transport, **kw)

    return factory


@pytest.mark.asyncio
async def test_post_session_content_posts_json() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    settings = MagicMock(spec=Settings)
    settings.platform_internal_service_token = "tok"
    settings.platform_api_base_url = "http://test"
    sections = [
        {"id": "goal", "label": "Goal", "estimatedMinutes": 5, "type": "goal", "content": "c"},
    ]
    mem = [
        {
            "proposalType": "NewSemantic",
            "title": "t",
            "summary": "",
            "proposedChangeJson": "{}",
            "priority": 0,
        },
    ]
    patch_client = patch.object(
        activities.httpx,
        "AsyncClient",
        side_effect=_async_client_with_transport(transport),
    )
    with patch.object(activities, "get_settings", return_value=settings), patch_client:
        await activities.post_session_content("sess-1", sections, mem)
    assert "/sessions/sess-1/session-content" in captured["url"]
    assert captured["body"]["sections"][0]["id"] == "goal"
    assert captured["body"]["memoryProposals"][0]["proposalType"] == "NewSemantic"


@pytest.mark.asyncio
async def test_generate_learning_session_parses_llm_json() -> None:
    llm_json = {
        "sections": [
            {
                "id": "goal",
                "label": "Goal",
                "estimatedMinutes": 5,
                "type": "goal",
                "content": "c1",
                "example": "e1",
            },
            {
                "id": "context",
                "label": "Context",
                "estimatedMinutes": 10,
                "type": "context",
                "content": "c2",
                "youtubeQuery": "q",
            },
            {
                "id": "hands-on",
                "label": "Hands-on",
                "estimatedMinutes": 30,
                "type": "hands-on",
                "content": "c3",
                "outputType": "code",
            },
            {
                "id": "reflection",
                "label": "Reflection",
                "estimatedMinutes": 10,
                "type": "reflection",
                "content": "c4",
                "prompts": ["a", "b", "c"],
            },
        ],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(llm_json)}}]},
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
        out = await activities.generate_learning_session(ctx, "Topic", None)
    assert len(out) == 4
    assert out[0]["id"] == "goal"


@pytest.mark.asyncio
async def test_generate_context_section_returns_plain_markdown() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert "response_format" not in body
        assert body.get("temperature") == 0.6
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "## Intro\n\nDeep teaching here."}}]},
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
        out = await activities.generate_context_section(ctx, "Topic", None)
    assert out.startswith("## Intro")
    assert "Deep teaching" in out


@pytest.mark.asyncio
async def test_analyze_topic_selection_for_memory_returns_wired() -> None:
    proposals = TopicSelectionMemoryProposalsLlmResponse(
        proposals=[
            TopicSelectionMemoryProposalLlmItem(
                proposal_type="NewSemantic",
                title="Interest",
                summary="User chose Rust",
                proposed_change={
                    "kind": "NewSemantic",
                    "key": "interest.topic",
                    "claim": "User is studying Rust ownership",
                    "domain": "learning",
                    "initialConfidence": 0.6,
                },
            ),
        ],
    )

    payload = proposals.model_dump(mode="json", by_alias=True)

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
        out = await activities.analyze_topic_selection_for_memory(ctx, "Rust", "more code")
    assert len(out) == 1
    assert out[0]["title"] == "Interest"


@pytest.mark.asyncio
async def test_fetch_memory_context_for_session_generation_calls_client() -> None:
    ctx = MemoryContextV1(profile_facts=[])
    mock_client = MagicMock()
    mock_client.post_memory_context = AsyncMock(return_value=ctx)
    with patch.object(activities, "get_settings") as gs:
        gs.return_value.consolidation_primary_user_id = 1
        with patch.object(activities, "PlatformMemoryHttpClient", return_value=mock_client):
            out = await activities.fetch_memory_context_for_session_generation(
                "My topic",
                "feedback",
            )
    mock_client.post_memory_context.assert_awaited_once()
    body = mock_client.post_memory_context.await_args.args[0]
    assert body.task_description == "My topic"
    assert out["profileFacts"] == []
