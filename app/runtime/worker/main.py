"""Temporal worker process entrypoint."""

import asyncio
import logging

from temporalio.worker import Worker

from app.runtime.config import get_settings
from app.runtime.logging import configure_logging
from app.runtime.registry import get_memory_consolidation_definitions, get_registered_definitions
from app.runtime.temporal import get_temporal_client

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Start and run Temporal workers (platform queue + memory-consolidation queue)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    platform_definitions = get_registered_definitions()
    memory_definitions = get_memory_consolidation_definitions()
    client = await get_temporal_client(settings)

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
    await asyncio.gather(platform_worker.run(), memory_worker.run())


def main() -> None:
    """Synchronous entrypoint for CLI execution."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
