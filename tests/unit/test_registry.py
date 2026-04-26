"""Unit tests for runtime workflow/activity registry."""

from app.runtime.registry import get_registered_definitions


def test_registry_returns_expected_workflow_and_activity_sets() -> None:
    """Registry should include both initial workflow modules."""
    definitions = get_registered_definitions()

    workflow_names = {workflow_cls.__name__ for workflow_cls in definitions.workflows}
    activity_names = {activity_fn.__name__ for activity_fn in definitions.activities}

    assert workflow_names == {"NewsIntelligenceWorkflow", "SideLearningWorkflow"}
    assert activity_names == {"fetch_news_sources", "collect_learning_candidates"}
