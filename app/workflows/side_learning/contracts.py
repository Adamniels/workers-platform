"""Typed contracts for side-learning workflow stages."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.workflow_contracts import WorkflowRunRequest


class SideLearningStage(StrEnum):
    PROPOSE_TOPICS = "propose_topics"
    GENERATE_SESSION = "generate_session"
    ANALYZE_REFLECTION = "analyze_reflection"


class SideLearningWorkflowRequest(WorkflowRunRequest):
    """Typed request for the side learning workflow."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str | None = Field(default=None, alias="sessionId")
    initial_prompt: str | None = Field(default=None, alias="initialPrompt")
    topic_title: str | None = Field(default=None, alias="topicTitle")
    user_feedback: str | None = Field(default=None, alias="userFeedback")
    reflection_text: str | None = Field(default=None, alias="reflectionText")
    session_content_json: str | None = Field(default=None, alias="sessionContentJson")


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


# --- Stage B: session content (four fixed sections) ---

EXPECTED_SECTION_IDS: tuple[str, str, str, str] = ("goal", "context", "hands-on", "reflection")


class SideLearningSessionSection(BaseModel):
    """One section in session content; aligns with implementation plan schema."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    label: str = ""
    estimated_minutes: int = Field(default=15, alias="estimatedMinutes")
    type: str = ""
    content: str = ""
    example: str | None = None
    youtube_query: str | None = Field(default=None, alias="youtubeQuery")
    output_type: str | None = Field(default=None, alias="outputType")
    prompts: list[str] | None = None


class SessionContentLlmResponse(BaseModel):
    """Structured JSON from the session-generation LLM."""

    model_config = ConfigDict(extra="ignore")

    sections: list[SideLearningSessionSection] = Field(default_factory=list)


# --- Stage B: memory proposals (review queue; MVP: NewSemantic + NewProceduralRule) ---


class TopicSelectionMemoryProposalLlmItem(BaseModel):
    """LLM output item before wire serialization and validation."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    proposal_type: str = Field(alias="proposalType")
    title: str = ""
    summary: str = ""
    proposed_change: dict[str, Any] = Field(default_factory=dict, alias="proposedChange")
    evidence: Any = None
    priority: int = 0


class TopicSelectionMemoryProposalsLlmResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    proposals: list[TopicSelectionMemoryProposalLlmItem] = Field(default_factory=list)


class SideLearningMemoryProposalWire(BaseModel):
    """Payload element for POST .../session-content memoryProposals (camelCase on wire)."""

    model_config = ConfigDict(populate_by_name=True)

    proposal_type: str = Field(alias="proposalType")
    title: str
    summary: str = ""
    proposed_change_json: str = Field(alias="proposedChangeJson")
    evidence_json: str | None = Field(default=None, alias="evidenceJson")
    priority: int = 0
