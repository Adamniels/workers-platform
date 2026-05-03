"""Cross-boundary workflow schemas mirrored for workers runtime."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowRunRequest(BaseModel):
    """Typed input contract used by worker workflows (JSON string payload from .NET)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    workflow_type: str = Field(alias="workflowType")
    task_queue: str | None = Field(default=None, alias="taskQueue")
    workflow_run_id: str = Field(alias="workflowRunId")
    stage: str | None = None
    session_id: str | None = Field(default=None, alias="sessionId")
    initial_prompt: str | None = Field(default=None, alias="initialPrompt")
    topic_title: str | None = Field(default=None, alias="topicTitle")
    user_feedback: str | None = Field(default=None, alias="userFeedback")
    reflection_text: str | None = Field(default=None, alias="reflectionText")
    session_content_json: str | None = Field(default=None, alias="sessionContentJson")


class WorkflowRunResult(BaseModel):
    """Typed workflow result payload for downstream artifact handling."""

    workflow_type: str = Field(alias="workflowType")
    workflow_run_id: str = Field(alias="workflowRunId")
    status: Literal["completed", "failed", "needs_input"]
    artifact_refs: list[str] = Field(default_factory=list, alias="artifactRefs")
