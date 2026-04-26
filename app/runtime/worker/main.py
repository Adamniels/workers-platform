"""Temporal worker process entrypoint."""

import asyncio
import logging

from temporalio.worker import Worker

from app.runtime.config import get_settings
from app.runtime.logging import configure_logging
from app.runtime.registry import get_registered_definitions
from app.runtime.temporal import get_temporal_client

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Start and run the Temporal worker process."""
    settings = get_settings()
    configure_logging(settings.log_level)

    definitions = get_registered_definitions()
    client = await get_temporal_client(settings)

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=list(definitions.workflows),
        activities=list(definitions.activities),
    )

    logger.info("starting worker on task queue %s", settings.temporal_task_queue)
    await worker.run()


def main() -> None:
    """Synchronous entrypoint for CLI execution."""
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
