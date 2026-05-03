"""Unit tests for runtime workflow/activity registry."""

from app.runtime.registry import get_memory_consolidation_definitions, get_registered_definitions


def test_registry_returns_expected_workflow_and_activity_sets() -> None:
    """Registry should include both initial workflow modules."""
    definitions = get_registered_definitions()

    workflow_names = {workflow_cls.__name__ for workflow_cls in definitions.workflows}
    activity_names = {activity_fn.__name__ for activity_fn in definitions.activities}

    assert workflow_names == {"NewsIntelligenceWorkflow", "SideLearningWorkflow"}
    assert activity_names == {
        "fetch_news_sources",
        "fetch_memory_context_for_learning",
        "fetch_memory_context_for_session_generation",
        "fetch_memory_context_for_reflection",
        "propose_learning_topics",
        "filter_known_topics",
        "post_topic_proposals",
        "generate_learning_session",
        "analyze_topic_selection_for_memory",
        "post_session_content",
        "analyze_reflection",
        "post_reflection_insights",
    }


def test_memory_consolidation_registry() -> None:
    definitions = get_memory_consolidation_definitions()
    assert {w.__name__ for w in definitions.workflows} == {"MemoryConsolidationWorkflow"}
    assert {a.__name__ for a in definitions.activities} == {
        "run_nightly_memory_consolidation_activity",
    }
