"""Temporal client and queue wiring."""

from .client import get_temporal_client
from .task_queues import DEFAULT_TASK_QUEUE

__all__ = ["DEFAULT_TASK_QUEUE", "get_temporal_client"]
