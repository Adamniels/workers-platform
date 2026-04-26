"""Adapter interface for memory service access."""

from typing import Protocol


class MemoryClient(Protocol):
    """Workflow-facing memory client contract."""

    async def fetch_context(self, user_id: str, query: str) -> list[str]:
        """Return memory context snippets for workflow reasoning."""
