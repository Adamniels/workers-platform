"""Workflow definition for side learning."""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from app.schemas.workflow_contracts import WorkflowRunRequest, WorkflowRunResult
from app.workflows.side_learning.contracts import SideLearningStage


@workflow.defn
class SideLearningWorkflow:
    """Workflow-owned AI execution for side learning."""

    @workflow.run
    async def run(self, payload: str) -> WorkflowRunResult:
        """Dispatch by stage: propose_topics (A), generate_session (B), reflection (C) TBD."""
        request = WorkflowRunRequest.model_validate_json(payload)
        stage = (request.stage or "").strip()

        if stage in (SideLearningStage.PROPOSE_TOPICS.value,):
            if not request.session_id:
                return WorkflowRunResult(
                    workflow_type=request.workflow_type,
                    workflow_run_id=request.workflow_run_id,
                    status="failed",
                    artifact_refs=["missing_session_id"],
                )

            ctx = await workflow.execute_activity(
                "fetch_memory_context_for_learning",
                request.initial_prompt,
                start_to_close_timeout=timedelta(seconds=60),
            )
            proposals = await workflow.execute_activity(
                "propose_learning_topics",
                args=[ctx, request.initial_prompt],
                start_to_close_timeout=timedelta(seconds=120),
            )
            filtered = await workflow.execute_activity(
                "filter_known_topics",
                args=[proposals, ctx],
                start_to_close_timeout=timedelta(seconds=30),
            )
            await workflow.execute_activity(
                "post_topic_proposals",
                args=[request.session_id, filtered],
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            titles = [str(t.get("title", "")) for t in filtered if isinstance(t, dict)]
            return WorkflowRunResult(
                workflow_type=request.workflow_type,
                workflow_run_id=request.workflow_run_id,
                status="completed",
                artifact_refs=titles,
            )

        if stage in (SideLearningStage.GENERATE_SESSION.value,):
            if not request.session_id or not (request.topic_title or "").strip():
                return WorkflowRunResult(
                    workflow_type=request.workflow_type,
                    workflow_run_id=request.workflow_run_id,
                    status="failed",
                    artifact_refs=["missing_session_id_or_topic_title"],
                )

            ctx = await workflow.execute_activity(
                "fetch_memory_context_for_session_generation",
                args=[request.topic_title, request.user_feedback],
                start_to_close_timeout=timedelta(seconds=60),
            )
            sections = await workflow.execute_activity(
                "generate_learning_session",
                args=[ctx, request.topic_title, request.user_feedback],
                start_to_close_timeout=timedelta(seconds=300),
            )
            memory_proposals = await workflow.execute_activity(
                "analyze_topic_selection_for_memory",
                args=[ctx, request.topic_title, request.user_feedback],
                start_to_close_timeout=timedelta(seconds=120),
            )
            await workflow.execute_activity(
                "post_session_content",
                args=[request.session_id, sections, memory_proposals],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            refs = [str(s.get("id", "")) for s in sections if isinstance(s, dict)]
            return WorkflowRunResult(
                workflow_type=request.workflow_type,
                workflow_run_id=request.workflow_run_id,
                status="completed",
                artifact_refs=refs,
            )

        if stage in (SideLearningStage.ANALYZE_REFLECTION.value,):
            return WorkflowRunResult(
                workflow_type=request.workflow_type,
                workflow_run_id=request.workflow_run_id,
                status="failed",
                artifact_refs=[f"stage_not_implemented:{stage}"],
            )

        return WorkflowRunResult(
            workflow_type=request.workflow_type,
            workflow_run_id=request.workflow_run_id,
            status="failed",
            artifact_refs=[f"unknown_stage:{stage or 'empty'}"],
        )
