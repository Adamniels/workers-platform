"""Runtime registry for workflow and activity definitions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.workflows.memory_consolidation.activities import (
    run_nightly_memory_consolidation_activity,
)
from app.workflows.memory_consolidation.workflow import MemoryConsolidationWorkflow
from app.workflows.news_intelligence.activities import (
    embed_news_articles,
    ensure_user_news_profile,
    fetch_arxiv_articles,
    fetch_gnews_articles,
    fetch_hacker_news_articles,
    fetch_rss_articles,
    ingest_articles,
    update_user_news_active_context,
    update_user_news_profile,
)
from app.workflows.news_intelligence.workflow import NewsIntelligenceWorkflow
from app.workflows.side_learning.activities import (
    analyze_reflection,
    analyze_topic_selection_for_memory,
    fetch_memory_context_for_learning,
    fetch_memory_context_for_reflection,
    fetch_memory_context_for_session_generation,
    filter_known_topics,
    generate_context_section,
    generate_learning_session,
    post_reflection_insights,
    post_session_content,
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
            fetch_rss_articles,
            fetch_hacker_news_articles,
            fetch_gnews_articles,
            fetch_arxiv_articles,
            ingest_articles,
            embed_news_articles,
            ensure_user_news_profile,
            update_user_news_profile,
            update_user_news_active_context,
            fetch_memory_context_for_learning,
            fetch_memory_context_for_session_generation,
            propose_learning_topics,
            filter_known_topics,
            post_topic_proposals,
            generate_learning_session,
            generate_context_section,
            analyze_topic_selection_for_memory,
            post_session_content,
            fetch_memory_context_for_reflection,
            analyze_reflection,
            post_reflection_insights,
        ],
    )


def get_memory_consolidation_definitions() -> WorkflowDefinitions:
    """Dedicated task queue for memory consolidation (see docs/memory/10-nightly-worker.md)."""
    return WorkflowDefinitions(
        workflows=[MemoryConsolidationWorkflow],
        activities=[run_nightly_memory_consolidation_activity],
    )
