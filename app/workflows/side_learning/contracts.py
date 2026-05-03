"""Typed contracts for side-learning workflow stages."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SideLearningStage(StrEnum):
    PROPOSE_TOPICS = "propose_topics"
    GENERATE_SESSION = "generate_session"
    ANALYZE_REFLECTION = "analyze_reflection"


class TopicProposalItem(BaseModel):
    """One topic candidate aligned with .NET internal proposals schema."""

    model_config = ConfigDict(populate_by_name=True)

    title: str
    rationale: str = ""
    estimated_minutes: int = Field(default=0, alias="estimatedMinutes")
    difficulty: str = ""
    target_skill_gap: str = Field(default="", alias="targetSkillGap")


class TopicProposalLlmResponse(BaseModel):
    """Structured JSON from the LLM."""

    model_config = ConfigDict(extra="ignore")

    topics: list[TopicProposalItem] = Field(default_factory=list)
