"""Temporal workflow: nightly memory consolidation (v1)."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class MemoryConsolidationWorkflow:
    """Single-step workflow; schedule in Temporal to run nightly."""

    @workflow.run
    async def run(self) -> dict:
        return await workflow.execute_activity(
            "run_nightly_memory_consolidation_activity",
            start_to_close_timeout=timedelta(minutes=6),
        )
