"""Cross-boundary workflow schemas mirrored for workers runtime."""

from typing import Literal

from pydantic import BaseModel, Field


class WorkflowRunRequest(BaseModel):
    """Typed input contract used by worker workflows."""

    name: str
    workflow_type: str = Field(alias="workflowType")
    task_queue: str | None = Field(default=None, alias="taskQueue")
    workflow_run_id: str = Field(alias="workflowRunId")


class WorkflowRunResult(BaseModel):
    """Typed workflow result payload for downstream artifact handling."""

    workflow_type: str = Field(alias="workflowType")
    workflow_run_id: str = Field(alias="workflowRunId")
    status: Literal["completed", "failed", "needs_input"]
    artifact_refs: list[str] = Field(default_factory=list, alias="artifactRefs")
