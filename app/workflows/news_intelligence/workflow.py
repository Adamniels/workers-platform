"""Workflow definition for news intelligence."""

from datetime import timedelta

from temporalio import workflow

from app.schemas.workflow_contracts import WorkflowRunRequest, WorkflowRunResult


@workflow.defn
class NewsIntelligenceWorkflow:
    """Workflow-owned AI execution for news intelligence."""

    @workflow.run
    async def run(self, payload: str) -> WorkflowRunResult:
        """Execute news intelligence flow with workflow-local judgment."""
        request = WorkflowRunRequest.model_validate_json(payload)
        sources = await workflow.execute_activity(
            "fetch_news_sources",
            request.name,
            start_to_close_timeout=timedelta(seconds=30),
        )
        return WorkflowRunResult(
            workflow_type=request.workflow_type,
            workflow_run_id=request.workflow_run_id,
            status="completed",
            artifact_refs=sources,
        )
