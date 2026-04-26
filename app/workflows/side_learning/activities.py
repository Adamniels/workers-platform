"""Activities for the side learning workflow."""

from temporalio import activity


@activity.defn
async def collect_learning_candidates(topic: str) -> list[str]:
    """Placeholder for side-learning candidate generation."""
    return [f"placeholder-learning-item-for:{topic}"]
