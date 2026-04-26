"""Activities for the news intelligence workflow."""

from temporalio import activity


@activity.defn
async def fetch_news_sources(topic: str) -> list[str]:
    """Placeholder for source collection logic."""
    return [f"placeholder-source-for:{topic}"]
