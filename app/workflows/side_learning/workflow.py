"""Workflow definition for side learning."""

from datetime import timedelta

from temporalio import workflow

from app.schemas.workflow_contracts import WorkflowRunRequest, WorkflowRunResult


@workflow.defn
class SideLearningWorkflow:
    """Workflow-owned AI execution for side learning."""

    @workflow.run
    async def run(self, payload: str) -> WorkflowRunResult:
        """Execute side-learning flow with workflow-local judgment."""
        request = WorkflowRunRequest.model_validate_json(payload)
        candidates = await workflow.execute_activity(
            "collect_learning_candidates",
            request.name,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return WorkflowRunResult(
            workflow_type=request.workflow_type,
            workflow_run_id=request.workflow_run_id,
            status="completed",
            artifact_refs=candidates,
        )
