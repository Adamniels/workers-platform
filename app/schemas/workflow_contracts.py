"""Cross-boundary workflow schemas mirrored for workers runtime."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkflowRunRequest(BaseModel):
    """Base fields present in every workflow payload (JSON string from .NET)."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    workflow_type: str = Field(alias="workflowType")
    task_queue: str | None = Field(default=None, alias="taskQueue")
    workflow_run_id: str = Field(alias="workflowRunId")
    stage: str | None = None


class WorkflowRunResult(BaseModel):
    """Typed workflow result payload for downstream artifact handling."""

    model_config = ConfigDict(populate_by_name=True)

    workflow_type: str = Field(alias="workflowType")
    workflow_run_id: str = Field(alias="workflowRunId")
    status: Literal["completed", "failed", "needs_input"]
    artifact_refs: list[str] = Field(default_factory=list, alias="artifactRefs")
