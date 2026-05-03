"""Prompt text and dedupe helpers for side-learning Stage A."""

from __future__ import annotations

import re

from app.memory.client.models import MemoryContextV1
from app.workflows.side_learning.contracts import TopicProposalItem


def build_topic_proposal_system_prompt() -> str:
    return (
        "You propose short learning topics for a single user. "
        "Return strict JSON only, no markdown, matching the schema given in the user message. "
        "Topics should be concrete, varied, and suitable for a self-paced session."
    )


def build_topic_proposal_user_prompt(context: MemoryContextV1, initial_prompt: str | None) -> str:
    lines: list[str] = []
    if initial_prompt and initial_prompt.strip():
        lines.append(f"User hint (optional): {initial_prompt.strip()}")
    else:
        lines.append("No user hint was provided; infer from profile and memory only.")

    if context.profile_facts:
        lines.append("\n## Profile signals")
        for p in context.profile_facts[:24]:
            lines.append(f"- ({p.source}) {p.text}")

    if context.active_goals:
        lines.append("\n## Active goals")
        for g in context.active_goals[:16]:
            lines.append(f"- {g.goal}")

    if context.relevant_projects:
        lines.append("\n## Projects")
        for pr in context.relevant_projects[:12]:
            ext = f" ({pr.external_id})" if pr.external_id else ""
            lines.append(f"- {pr.name}{ext}")

    learning_semantics = [
        s for s in context.semantic_memories if (s.domain or "").lower() == "learning"
    ]
    other_semantics = [
        s for s in context.semantic_memories if (s.domain or "").lower() != "learning"
    ]
    if learning_semantics:
        lines.append("\n## Semantic memories (domain=learning)")
        for s in learning_semantics[:20]:
            lines.append(f"- [{s.key}] {s.claim}")
    if other_semantics[:10]:
        lines.append("\n## Other semantic memories (top)")
        for s in other_semantics[:10]:
            lines.append(f"- [{s.key}] {s.claim}")

    rules = [
        r
        for r in context.procedural_rules
        if (r.workflow_type or "").lower() == "side_learning"
    ]
    if rules:
        lines.append("\n## Procedural rules (workflowType=side_learning)")
        for r in rules[:16]:
            lines.append(f"- {r.rule_name}: {r.rule_content}")

    doc_recalls = [
        r
        for r in context.memory_item_vector_recalls
        if r.is_document_evidence or (r.memory_type or "").lower() == "document"
    ]
    if doc_recalls:
        lines.append("\n## Recent document-like memory hits (avoid duplicating these titles)")
        for r in doc_recalls[:20]:
            lines.append(f"- {r.title}: {r.content_preview[:200]}")

    lines.append(
        '\n## Output schema\n'
        'Return JSON: {"topics":[{"title":"...","rationale":"...","estimatedMinutes":30,'
        '"difficulty":"beginner|intermediate|advanced","targetSkillGap":"..."}]} '
        "with exactly 5 topics. estimatedMinutes must be a positive integer."
    )
    return "\n".join(lines)


def _normalize_title(s: str) -> str:
    s = s.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _title_too_similar(topic_title: str, recall_title: str) -> bool:
    """Strict match: substring containment after normalize; min length gate."""
    a = _normalize_title(topic_title)
    b = _normalize_title(recall_title)
    if len(a) < 4 or len(b) < 4:
        return False
    return a in b or b in a


def _document_recall_titles(ctx: MemoryContextV1) -> list[str]:
    titles: list[str] = []
    for r in ctx.memory_item_vector_recalls:
        if r.is_document_evidence or (r.memory_type or "").lower() == "document":
            if r.title:
                titles.append(r.title)
    return titles


def filter_proposals_against_recalls(
    proposals: list[TopicProposalItem],
    context: MemoryContextV1,
) -> list[TopicProposalItem]:
    """
    Remove topics whose title is too close to a document vector-recall title.

    If fewer than 3 remain after strict filtering, relax to substring-only on long titles,
    then if still <3 return the original list (prefer shipping topics over blocking the flow).
    """
    if len(proposals) <= 3:
        return proposals

    recall_titles = _document_recall_titles(context)

    def apply_filter(strict: bool) -> list[TopicProposalItem]:
        kept: list[TopicProposalItem] = []
        for p in proposals:
            drop = False
            for rt in recall_titles:
                if strict and _title_too_similar(p.title, rt):
                    drop = True
                    break
                if not strict:
                    nt = _normalize_title(p.title)
                    nrt = _normalize_title(rt)
                    if len(nt) >= 8 and nt in nrt:
                        drop = True
                        break
            if not drop:
                kept.append(p)
        return kept

    strict_kept = apply_filter(strict=True)
    if len(strict_kept) >= 3:
        return strict_kept

    relaxed_kept = apply_filter(strict=False)
    if len(relaxed_kept) >= 3:
        return relaxed_kept

    # Not enough signal to filter safely — keep original proposals (>=3 by contract from LLM).
    return proposals
