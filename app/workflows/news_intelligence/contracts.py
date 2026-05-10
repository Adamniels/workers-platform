"""Pydantic contracts for news ingestion (worker ↔ .NET internal API)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ArticleCandidate(BaseModel):
    """Normalized article from any source before dedupe + ingest."""

    model_config = ConfigDict(populate_by_name=True)

    title: str
    url: str
    source: str
    body: str = ""
    author: str | None = None
    published_at: datetime = Field(alias="publishedAt")
    source_feed_url: str | None = Field(default=None, alias="sourceFeedUrl")


class NewsIngestResult(BaseModel):
    """Aggregate result from ingest_articles activity."""

    model_config = ConfigDict(populate_by_name=True)

    created: int = 0
    duplicates: int = 0
    errors: int = 0


class IngestNewsItemV1Request(BaseModel):
    """JSON body for POST /api/internal/v1/news/items."""

    model_config = ConfigDict(populate_by_name=True)

    title: str
    url: str
    source: str
    body: str
    author: str | None = None
    published_at: datetime = Field(alias="publishedAt")
    source_feed_url: str | None = Field(default=None, alias="sourceFeedUrl")


class IngestNewsItemV1Response(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: Literal["created", "duplicate"]
    id: str | None = Field(default=None, alias="id")
