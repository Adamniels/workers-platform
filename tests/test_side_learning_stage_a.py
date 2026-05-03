"""Unit tests for side-learning Stage A helpers and memory models."""

import json
from pathlib import Path

from app.memory.client.models import MemoryContextV1
from app.workflows.side_learning.contracts import TopicProposalItem
from app.workflows.side_learning.memory_mapper import filter_proposals_against_recalls


def _fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text())


def test_memory_context_v1_parses_fixture() -> None:
    data = _fixture("memory_context_min.json")
    ctx = MemoryContextV1.model_validate(data)
    assert ctx.assembly_stage == "v1-sql"
    assert len(ctx.profile_facts) == 1


def test_filter_drops_near_duplicate_document_title() -> None:
    ctx = MemoryContextV1(
        memory_item_vector_recalls=[
            {
                "memoryItemId": 1,
                "chunkIndex": 0,
                "memoryType": "Document",
                "title": "Side Learning: Temporal basics — 2026-01-01",
                "contentPreview": "notes",
                "cosineSimilarity": 0.9,
                "authorityWeight": 1,
                "rankScore": 0.5,
                "embeddingModelKey": "k",
                "isDocumentEvidence": True,
                "sourceType": "side_learning",
            }
        ],
    )
    proposals = [
        TopicProposalItem(
            title="Temporal basics",
            rationale="r",
            estimated_minutes=30,
            difficulty="easy",
        ),
        TopicProposalItem(
            title="Rust ownership",
            rationale="r2",
            estimated_minutes=45,
            difficulty="med",
        ),
        TopicProposalItem(
            title="Postgres indexing",
            rationale="r3",
            estimated_minutes=40,
            difficulty="med",
        ),
        TopicProposalItem(
            title="HTTP/2 internals",
            rationale="r4",
            estimated_minutes=50,
            difficulty="hard",
        ),
        TopicProposalItem(
            title="Testing strategies",
            rationale="r5",
            estimated_minutes=35,
            difficulty="easy",
        ),
    ]
    out = filter_proposals_against_recalls(proposals, ctx)
    titles = {p.title for p in out}
    assert "Temporal basics" not in titles
    assert len(out) >= 3
