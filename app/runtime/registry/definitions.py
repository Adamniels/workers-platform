"""Runtime registry for workflow and activity definitions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.workflows.news_intelligence.activities import fetch_news_sources
from app.workflows.news_intelligence.workflow import NewsIntelligenceWorkflow
from app.workflows.side_learning.activities import collect_learning_candidates
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
        activities=[fetch_news_sources, collect_learning_candidates],
    )
