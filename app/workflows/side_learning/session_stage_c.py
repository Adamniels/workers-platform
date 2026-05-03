"""Stage C prompts: reflection-driven memory proposals (LLM schema matches Stage B)."""

from __future__ import annotations

from app.memory.client.models import MemoryContextV1
from app.workflows.side_learning.session_stage_b import append_memory_context_narrative

_REFLECTION_SESSION_JSON_MAX = 12_000


def _trunc_session_json(session_content_json: str) -> str:
    s = (session_content_json or "").strip()
    if len(s) <= _REFLECTION_SESSION_JSON_MAX:
        return s
    return s[: _REFLECTION_SESSION_JSON_MAX - 24] + "\n... [truncated]"


def build_reflection_memory_system_prompt() -> str:
    return (
        "You infer durable memory candidates from a completed side-learning session and the "
        "user's written reflection. Return strict JSON only, no markdown. "
        "Only output proposals when there is clear signal from the reflection text; "
        "otherwise return an empty proposals array. "
        "proposalType must be exactly 'NewSemantic' or 'NewProceduralRule'."
    )


def build_reflection_memory_user_prompt(
    context: MemoryContextV1,
    topic_title: str,
    reflection_text: str,
    session_content_json: str,
) -> str:
    session_block = "\n## Session content (JSON, may be truncated)\n"
    session_block += _trunc_session_json(session_content_json)
    lines: list[str] = [
        f"Session topic: {topic_title.strip()}",
        f"User reflection:\n{(reflection_text or '').strip()}",
        session_block,
    ]
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
        "Use at most 5 proposals. Skip speculative or duplicate items."
    )
    return "\n".join(lines)
