"""Temporal worker process entrypoint."""

import asyncio
import json
import logging

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleUpdate,
)
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Worker

from app.runtime.config import get_settings
from app.runtime.logging import configure_logging
from app.runtime.registry import get_memory_consolidation_definitions, get_registered_definitions
from app.runtime.temporal import get_temporal_client

logger = logging.getLogger(__name__)

_NEWS_SCHEDULE_ID = "news-intelligence-weekly"
_NEWS_CRON = "0 22 * * 0"  # Sunday at 22:00 UTC

_MEMORY_SCHEDULE_ID = "memory-consolidation-nightly"
_MEMORY_CRON = "0 2 * * *"  # Every night at 02:00 UTC


async def _upsert_schedule(client: Client, schedule_id: str, schedule: Schedule) -> None:
    """Create the schedule if it does not exist; update it if it does."""
    try:
        await client.create_schedule(schedule_id, schedule)
        logger.info("schedule created id=%s", schedule_id)
    except RPCError as e:
        if e.status == RPCStatusCode.ALREADY_EXISTS:
            handle = client.get_schedule_handle(schedule_id)
            await handle.update(lambda _: ScheduleUpdate(schedule=schedule))
            logger.info("schedule updated id=%s", schedule_id)
        else:
            logger.warning("could not ensure schedule id=%s err=%s", schedule_id, e)


async def _ensure_schedules(client: Client, task_queue: str, memory_task_queue: str) -> None:
    """Upsert all Temporal schedules so they stay in sync with code on every deploy."""
    await _upsert_schedule(
        client,
        _NEWS_SCHEDULE_ID,
        Schedule(
            action=ScheduleActionStartWorkflow(
                "NewsIntelligenceWorkflow",
                json.dumps({"workflow_type": "news_intelligence", "workflow_run_id": "scheduled"}),
                id="news-intelligence-scheduled",
                task_queue=task_queue,
            ),
            spec=ScheduleSpec(cron_expressions=[_NEWS_CRON]),
            policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
        ),
    )
    await _upsert_schedule(
        client,
        _MEMORY_SCHEDULE_ID,
        Schedule(
            action=ScheduleActionStartWorkflow(
                "MemoryConsolidationWorkflow",
                id="memory-consolidation-scheduled",
                task_queue=memory_task_queue,
            ),
            spec=ScheduleSpec(cron_expressions=[_MEMORY_CRON]),
            policy=SchedulePolicy(overlap=ScheduleOverlapPolicy.SKIP),
        ),
    )


async def run_worker() -> None:
    """Start and run Temporal workers (platform queue + memory-consolidation queue)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    platform_definitions = get_registered_definitions()
    memory_definitions = get_memory_consolidation_definitions()
    client = await get_temporal_client(settings)

    await _ensure_schedules(client, settings.temporal_task_queue, settings.temporal_task_queue_memory)

    platform_worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=list(platform_definitions.workflows),
        activities=list(platform_definitions.activities),
    )
    memory_worker = Worker(
        client,
        task_queue=settings.temporal_task_queue_memory,
        workflows=list(memory_definitions.workflows),
        activities=list(memory_definitions.activities),
    )

    logger.info(
        "starting workers queues=%s,%s",
        settings.temporal_task_queue,
        settings.temporal_task_queue_memory,
    )
    try:
        await asyncio.gather(platform_worker.run(), memory_worker.run())
    except asyncio.CancelledError:
        # Ctrl+C / SIGTERM path: cancel is expected; re-raise would be correct inside
        # nested tasks, but at this top-level it only produces a noisy traceback.
        pass


def main() -> None:
    """Synchronous entrypoint for CLI execution."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
