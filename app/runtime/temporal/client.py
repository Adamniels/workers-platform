"""Temporal client factory."""

from temporalio.client import Client

from app.runtime.config import Settings


async def get_temporal_client(settings: Settings) -> Client:
    """Connect to Temporal using runtime settings."""
    return await Client.connect(
        target_host=settings.temporal_server_url,
        namespace=settings.temporal_namespace,
    )
