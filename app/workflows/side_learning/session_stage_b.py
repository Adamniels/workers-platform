"""Stage B prompts, session normalization, and memory proposal validation."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.memory.client.models import MemoryContextV1
from app.workflows.side_learning.contracts import (
    EXPECTED_SECTION_IDS,
    SideLearningMemoryProposalWire,
    SideLearningSessionSection,
    TopicSelectionMemoryProposalLlmItem,
)

logger = logging.getLogger(__name__)

_SECTION_DEFAULTS: tuple[tuple[str, str, str], ...] = (
    ("goal", "Goal", "goal"),
    ("context", "Context", "context"),
    ("hands-on", "Hands-on", "hands-on"),
    ("reflection", "Reflection", "reflection"),
)


def append_memory_context_narrative(lines: list[str], context: MemoryContextV1) -> None:
    if context.profile_facts:
        lines.append("\n## Profile signals")
        for p in context.profile_facts[:24]:
            lines.append(f"- ({p.source}) {p.text}")

    if context.active_goals:
        lines.append("\n## Active goals")
        for g in context.active_goals[:16]:
            lines.append(f"- {g.goal}")

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

    recalls = context.memory_item_vector_recalls[:18]
    if recalls:
        lines.append("\n## Vector recalls (documents and snippets)")
        for r in recalls:
            lines.append(f"- [{r.title}] {r.content_preview[:280]}")


def build_session_generation_system_prompt() -> str:
    return (
        "You design a single self-paced learning session for one user. "
        "Return strict JSON only, no markdown. "
        "You must output exactly four sections in order: goal, context, hands-on, reflection."
    )


def build_session_generation_user_prompt(
    context: MemoryContextV1,
    topic_title: str,
    user_feedback: str | None,
) -> str:
    lines: list[str] = [
        f"Chosen topic title: {topic_title.strip()}",
    ]
    if user_feedback and user_feedback.strip():
        lines.append(f"User feedback on the topic (tone, depth, focus): {user_feedback.strip()}")
    else:
        lines.append("No extra user feedback beyond the topic title.")

    append_memory_context_narrative(lines, context)

    lines.append(
        "\n## Output schema\n"
        'Return JSON: {"sections":[...]} with exactly 4 objects in this order of "id": '
        '"goal", "context", "hands-on", "reflection".\n'
        "Each section object fields:\n"
        "- id, label, estimatedMinutes (int), type (= id), content (plain text ok).\n"
        '- goal: include "example" (string).\n'
        '- context: include "youtubeQuery" (string) for video search.\n'
        '- hands-on: include "outputType" e.g. code|diagram|notes|demo.\n'
        '- reflection: include "prompts" (3–4 strings); last may fit the topic.\n'
        "estimatedMinutes should be positive and realistic for that section."
    )
    return "\n".join(lines)


def build_topic_memory_system_prompt() -> str:
    return (
        "You infer durable memory candidates from the chosen topic and optional feedback. "
        "Return strict JSON only, no markdown. "
        "Only output proposals when there is clear signal; else return empty proposals. "
        "proposalType must be exactly 'NewSemantic' or 'NewProceduralRule'."
    )


def build_topic_memory_user_prompt(
    context: MemoryContextV1,
    topic_title: str,
    user_feedback: str | None,
) -> str:
    lines: list[str] = [
        f"Chosen topic: {topic_title.strip()}",
    ]
    if user_feedback and user_feedback.strip():
        lines.append(f"User feedback: {user_feedback.strip()}")
    append_memory_context_narrative(lines, context)
    lines.append(
        '\n## Output schema\n'
        'Return JSON: {"proposals":[{"proposalType":"NewSemantic","title":"...","summary":"...",'
        '"proposedChange":{...},"evidence":null,"priority":0}]}\n'
        "For NewSemantic, proposedChange must be: "
        '{"kind":"NewSemantic","key":"...","claim":"...","domain":"learning|null",'
        '"initialConfidence":0.0-1.0}\n'
        "For NewProceduralRule, proposedChange must be: "
        '{"kind":"NewProceduralRule","workflowType":"side_learning","ruleName":"...",'
        '"ruleContent":"...","priority":0,"source":"side_learning_worker",'
        '"authorityWeight":0.55}\n'
        "Use at most 3 proposals. Skip speculative or duplicate items."
    )
    return "\n".join(lines)


def normalize_session_sections(sections: list[SideLearningSessionSection]) -> list[dict[str, Any]]:
    """Ensure four sections with expected ids; fill defaults; dump for API (camelCase)."""
    by_id = {s.id: s for s in sections}
    out: list[dict[str, Any]] = []
    default_prompts = [
        "What did I learn?",
        "What was confusing?",
        "What would I improve?",
    ]
    for sid, label, typ in _SECTION_DEFAULTS:
        raw = by_id.get(sid)
        if raw is None:
            em = 45 if sid == "hands-on" else 15
            sec = SideLearningSessionSection(
                id=sid,
                label=label,
                estimated_minutes=em,
                type=typ,
                content="",
                example="" if sid == "goal" else None,
                youtube_query="" if sid == "context" else None,
                output_type="code" if sid == "hands-on" else None,
                prompts=list(default_prompts) if sid == "reflection" else None,
            )
        else:
            em = raw.estimated_minutes if raw.estimated_minutes > 0 else 15
            if sid == "hands-on" and em < 30:
                em = 45
            sec_type = (raw.type or "").strip() or typ
            ex = raw.example if sid == "goal" else None
            if sid == "goal" and ex is None:
                ex = ""
            yq = raw.youtube_query if sid == "context" else None
            if sid == "context" and not (yq or "").strip():
                yq = ""
            ot = raw.output_type if sid == "hands-on" else None
            if sid == "hands-on" and not (ot or "").strip():
                ot = "code"
            pr = raw.prompts if sid == "reflection" else None
            if sid == "reflection":
                pr = (pr if pr else list(default_prompts))[:4]
            sec = SideLearningSessionSection(
                id=sid,
                label=raw.label.strip() or label,
                estimated_minutes=em,
                type=sec_type,
                content=raw.content.strip(),
                example=ex,
                youtube_query=yq,
                output_type=ot,
                prompts=pr,
            )

        dumped = sec.model_dump(mode="json", by_alias=True, exclude_none=False)
        if sid == "reflection" and not dumped.get("prompts"):
            dumped["prompts"] = list(default_prompts)
        out.append(dumped)

    ids = [s["id"] for s in out]
    if ids != list(EXPECTED_SECTION_IDS):
        logger.warning("normalize_session_sections: unexpected id order %s", ids)
    return out


def _evidence_json_blob(evidence: Any) -> str | None:
    if evidence is None:
        return None
    if isinstance(evidence, str):
        return evidence
    try:
        return json.dumps(evidence)
    except (TypeError, ValueError):
        return None


def _validate_new_semantic(pc: dict[str, Any]) -> dict[str, Any] | None:
    pc = {**pc, "kind": "NewSemantic"}
    key = str(pc.get("key", "")).strip()
    claim = str(pc.get("claim", "")).strip()
    if not key or not claim:
        return None
    domain = pc.get("domain")
    conf = pc.get("initialConfidence", 0.65)
    try:
        cf = float(conf)
    except (TypeError, ValueError):
        cf = 0.65
    cf = max(0.0, min(1.0, cf))
    out = {"kind": "NewSemantic", "key": key, "claim": claim, "initialConfidence": cf}
    if domain is not None and str(domain).strip():
        out["domain"] = str(domain).strip()
    return out


def _validate_new_procedural(pc: dict[str, Any]) -> dict[str, Any] | None:
    pc = {**pc, "kind": "NewProceduralRule"}
    content = str(pc.get("ruleContent", "")).strip()
    if not content:
        return None
    wf = str(pc.get("workflowType", "side_learning")).strip() or "side_learning"
    rn = str(pc.get("ruleName", "")).strip()
    if not rn:
        return None
    src = str(pc.get("source", "side_learning_worker")).strip() or "side_learning_worker"
    try:
        pr = int(pc.get("priority", 0))
    except (TypeError, ValueError):
        pr = 0
    try:
        aw = float(pc.get("authorityWeight", 0.55))
    except (TypeError, ValueError):
        aw = 0.55
    out: dict[str, Any] = {
        "kind": "NewProceduralRule",
        "workflowType": wf,
        "ruleName": rn,
        "ruleContent": content,
        "priority": pr,
        "source": src,
        "authorityWeight": aw,
    }
    basis = pc.get("basisRuleId")
    if basis is not None:
        try:
            bid = int(basis)
            if bid > 0:
                out["basisRuleId"] = bid
        except (TypeError, ValueError):
            pass
    return out


def try_wire_memory_proposal(
    item: TopicSelectionMemoryProposalLlmItem,
) -> SideLearningMemoryProposalWire | None:
    pt = (item.proposal_type or "").strip()
    pc_in = item.proposed_change
    if not isinstance(pc_in, dict):
        return None
    if pt == "NewSemantic":
        fixed = _validate_new_semantic(pc_in)
        if fixed is None:
            return None
        title = (item.title or "").strip() or str(fixed["key"])[:512]
        summary = (item.summary or "").strip()
        pj = json.dumps(fixed, separators=(",", ":"))
        return SideLearningMemoryProposalWire(
            proposal_type="NewSemantic",
            title=title[:512],
            summary=summary[:4000],
            proposed_change_json=pj,
            evidence_json=_evidence_json_blob(item.evidence),
            priority=max(0, min(1_000_000, int(item.priority))),
        )
    if pt == "NewProceduralRule":
        fixed = _validate_new_procedural(pc_in)
        if fixed is None:
            return None
        title = (item.title or "").strip() or str(fixed["ruleName"])[:512]
        summary = (item.summary or "").strip()
        pj = json.dumps(fixed, separators=(",", ":"))
        return SideLearningMemoryProposalWire(
            proposal_type="NewProceduralRule",
            title=title[:512],
            summary=summary[:4000],
            proposed_change_json=pj,
            evidence_json=_evidence_json_blob(item.evidence),
            priority=max(0, min(1_000_000, int(item.priority))),
        )
    return None


def wire_memory_proposals_from_llm(
    items: list[TopicSelectionMemoryProposalLlmItem],
    max_proposals: int = 3,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cap = max(0, min(max_proposals, 8))
    for it in items[:8]:
        w = try_wire_memory_proposal(it)
        if w is not None:
            out.append(w.model_dump(mode="json", by_alias=True, exclude_none=True))
        if len(out) >= cap:
            break
    return out[:cap]
