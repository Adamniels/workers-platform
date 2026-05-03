"""Runtime registry for workflow and activity definitions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.workflows.memory_consolidation.activities import (
    run_nightly_memory_consolidation_activity,
)
from app.workflows.memory_consolidation.workflow import MemoryConsolidationWorkflow
from app.workflows.news_intelligence.activities import fetch_news_sources
from app.workflows.news_intelligence.workflow import NewsIntelligenceWorkflow
from app.workflows.side_learning.activities import (
    fetch_memory_context_for_learning,
    filter_known_topics,
    post_topic_proposals,
    propose_learning_topics,
)
from app.workflows.side_learning.workflow import SideLearningWorkflow

ActivityFn = Callable[..., Any]


@dataclass(frozen=True)
class WorkflowDefinitions:
    """Container for worker-registered workflows and activities."""

    workflows: Sequence[type]
    activities: Sequence[ActivityFn]


def get_registered_definitions() -> WorkflowDefinitions:
    """Return all workflow and activity definitions to register in the worker."""
    return WorkflowDefinitions(
        workflows=[NewsIntelligenceWorkflow, SideLearningWorkflow],
        activities=[
            fetch_news_sources,
            fetch_memory_context_for_learning,
            propose_learning_topics,
            filter_known_topics,
            post_topic_proposals,
        ],
    )


def get_memory_consolidation_definitions() -> WorkflowDefinitions:
    """Dedicated task queue for memory consolidation (see docs/memory/10-nightly-worker.md)."""
    return WorkflowDefinitions(
        workflows=[MemoryConsolidationWorkflow],
        activities=[run_nightly_memory_consolidation_activity],
    )
