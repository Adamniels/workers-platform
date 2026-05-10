"""Temporal activities: fetch sources and ingest into Platform API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
import trafilatura
from bs4 import BeautifulSoup
from temporalio import activity

from app.runtime.config import get_settings
from app.workflows.news_intelligence.contracts import (
    ArticleCandidate,
    IngestNewsItemV1Request,
    IngestNewsItemV1Response,
    NewsIngestResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_RSS_FEEDS = [
    "https://hnrss.org/frontpage",
    "http://feeds.arstechnica.com/arstechnica/index/",
]
_DEFAULT_GNEWS_TOPICS = [
    "artificial intelligence",
    "software engineering",
    "climate",
    "space",
    "cybersecurity",
]
_DEFAULT_ARXIV_CATEGORIES = ["cs.AI", "cs.LG"]


_FULL_CONTENT_MIN_CHARS = 500  # below this we try trafilatura
_MAX_AGE_DAYS = 7              # discard articles older than this (all sources)


def _strip_html(raw: str) -> str:
    """Strip HTML tags and collapse excessive whitespace."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "lxml").get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_rss_body(entry: Any) -> str:
    """Read the richest content field available from a feedparser entry.

    Priority:
      1. entry.content  — feedparser's name for <content:encoded> / Atom <content>
         (full article body; most technical blogs include this)
      2. entry.summary / entry.description  — short excerpt, RSS <description>
    HTML is stripped from whichever field is used.
    """
    content_list = getattr(entry, "content", None) or []
    for c in content_list:
        val = getattr(c, "value", "") or ""
        if val.strip():
            return _strip_html(val)
    desc = getattr(entry, "description", None)
    raw = (getattr(entry, "summary", None) or desc or "").strip()
    return _strip_html(raw)


def _fetch_full_article(url: str) -> str:
    """Fetch and extract clean article text via trafilatura. Returns "" on failure."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        return (extracted or "").strip()
    except Exception:
        logger.warning("trafilatura fetch failed url=%s", url)
        return ""


def _parse_feed_date(entry: Any) -> datetime:
    if getattr(entry, "published_parsed", None):
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    if getattr(entry, "updated_parsed", None):
        try:
            return datetime(*entry.updated_parsed[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if raw:
        try:
            dt = parsedate_to_datetime(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except (TypeError, ValueError, OverflowError):
            pass
    return datetime.now(tz=UTC)


def _rss_feed_urls() -> list[str]:
    settings = get_settings()
    raw = (settings.news_rss_feed_urls_json or "").strip()
    if not raw:
        return list(_DEFAULT_RSS_FEEDS)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x and str(x).strip()]
    except json.JSONDecodeError:
        logger.warning("NEWS_RSS_FEED_URLS_JSON is not valid JSON; using defaults")
    return list(_DEFAULT_RSS_FEEDS)


def _article_json(articles: list[ArticleCandidate]) -> list[dict]:
    return [a.model_dump(mode="json", by_alias=True) for a in articles]


@activity.defn
async def fetch_rss_articles(feed_urls: list[str] | None = None) -> list[dict]:
    urls = feed_urls if feed_urls else _rss_feed_urls()
    out: list[ArticleCandidate] = []

    def parse_one(url: str) -> list[ArticleCandidate]:
        local: list[ArticleCandidate] = []
        try:
            parsed = feedparser.parse(url)
        except Exception:
            logger.exception("RSS parse failed url=%s", url)
            return local

        feed_title = (getattr(parsed.feed, "title", None) or url)[:256]
        cutoff = datetime.now(tz=UTC) - timedelta(days=_MAX_AGE_DAYS)

        for entry in getattr(parsed, "entries", []) or []:
            title = (getattr(entry, "title", None) or "").strip()
            link = (getattr(entry, "link", None) or "").strip()
            if not title or not link:
                logger.warning("RSS skip entry without title/url feed=%s", url)
                continue

            pub = _parse_feed_date(entry)
            if pub < cutoff:
                logger.debug("RSS skip old article published=%s url=%s", pub.date(), link)
                continue

            # Prefer content:encoded / Atom <content> (full body); fall back to summary
            body = _extract_rss_body(entry)

            # If the feed only ships a short excerpt, fetch the full article
            if len(body) < _FULL_CONTENT_MIN_CHARS:
                fetched = _fetch_full_article(link)
                if len(fetched) > len(body):
                    body = fetched
                    logger.debug("trafilatura enriched url=%s chars=%d", link, len(body))

            author = getattr(entry, "author", None)
            if author:
                author = str(author).strip() or None

            local.append(
                ArticleCandidate(
                    title=title,
                    url=link,
                    source=feed_title,
                    body=body[:500_000],
                    author=author,
                    published_at=pub,
                    source_feed_url=url,
                )
            )
        return local

    for u in urls:
        try:
            batch = await asyncio.to_thread(parse_one, u)
            out.extend(batch)
            logger.info("RSS feed fetched url=%s articles=%d", u, len(batch))
        except Exception:
            logger.exception("RSS feed failed url=%s", u)
    return _article_json(out)


@activity.defn
async def fetch_hacker_news_articles(min_score: int = 100, max_results: int = 30) -> list[dict]:
    url = (
        "https://hn.algolia.com/api/v1/search"
        f"?tags=story&numericFilters=points%3E{min_score}&hitsPerPage={max_results}"
    )
    out: list[ArticleCandidate] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        logger.exception("Hacker News API request failed")
        return []

    cutoff = datetime.now(tz=UTC) - timedelta(days=_MAX_AGE_DAYS)

    for hit in data.get("hits", []) or []:
        title = (hit.get("title") or "").strip()
        story_url = (hit.get("url") or "").strip() or None
        if not story_url:
            continue
        # Skip job posts — they have no editorial value as news
        if hit.get("_tags") and "job" in hit.get("_tags", []):
            continue
        created = hit.get("created_at")
        try:
            pub = datetime.fromisoformat(str(created).replace("Z", "+00:00")) if created else datetime.now(tz=UTC)
        except (TypeError, ValueError):
            pub = datetime.now(tz=UTC)
        if pub < cutoff:
            continue
        text = (hit.get("story_text") or "").strip()
        summary = text[:8000] if text else ""
        out.append(
            ArticleCandidate(
                title=title or "Untitled",
                url=story_url,
                source="Hacker News",
                body=summary[:500_000],
                author=None,
                published_at=pub,
                source_feed_url=None,
            )
        )
    return _article_json(out)


def _gnews_topics() -> list[str]:
    settings = get_settings()
    raw = (settings.news_gnews_topics_json or "").strip()
    if not raw:
        return list(_DEFAULT_GNEWS_TOPICS)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x and str(x).strip()]
    except json.JSONDecodeError:
        logger.warning("NEWS_GNEWS_TOPICS_JSON invalid; using defaults")
    return list(_DEFAULT_GNEWS_TOPICS)


@activity.defn
async def fetch_gnews_articles(
    topics: list[str] | None = None,
    max_per_topic: int = 5,
) -> list[dict]:
    settings = get_settings()
    key = (settings.gnews_api_key or "").strip()
    if not key:
        logger.warning("GNEWS_API_KEY not set; skipping GNews")
        return []
    topic_list = topics if topics else _gnews_topics()
    out: list[ArticleCandidate] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for topic in topic_list:
            url = "https://gnews.io/api/v4/search"
            try:
                await asyncio.sleep(0.5)
                r = await client.get(
                    url,
                    params={
                        "q": topic,
                        "max": str(max_per_topic),
                        "token": key,
                        "lang": "en",
                    },
                )
                if r.status_code == 429:
                    logger.warning("GNews rate limited topic=%s", topic)
                    continue
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPStatusError as ex:
                if ex.response is not None and ex.response.status_code == 429:
                    logger.warning("GNews 429 topic=%s", topic)
                    continue
                logger.exception("GNews HTTP error topic=%s", topic)
                raise
            except httpx.HTTPError:
                logger.exception("GNews request failed topic=%s", topic)
                continue
            for art in payload.get("articles", []) or []:
                t = (art.get("title") or "").strip()
                u = (art.get("url") or "").strip()
                if not t or not u:
                    continue
                desc = (art.get("description") or "").strip()
                src = (art.get("source") or {}) if isinstance(art.get("source"), dict) else {}
                src_name = (src.get("name") or "GNews")[:256]
                pub_raw = art.get("publishedAt")
                try:
                    if pub_raw:
                        pub = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00"))
                    else:
                        pub = datetime.now(tz=UTC)
                except (TypeError, ValueError):
                    pub = datetime.now(tz=UTC)
                out.append(
                    ArticleCandidate(
                        title=t,
                        url=u,
                        source=src_name,
                        body=desc[:500_000],
                        author=None,
                        published_at=pub,
                        source_feed_url=None,
                    )
                )
    return _article_json(out)


def _arxiv_categories() -> list[str]:
    settings = get_settings()
    raw = (settings.news_arxiv_categories_json or "").strip()
    if not raw:
        return list(_DEFAULT_ARXIV_CATEGORIES)
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x and str(x).strip()]
    except json.JSONDecodeError:
        logger.warning("NEWS_ARXIV_CATEGORIES_JSON invalid; using defaults")
    return list(_DEFAULT_ARXIV_CATEGORIES)


@activity.defn
async def fetch_arxiv_articles(categories: list[str] | None = None) -> list[dict]:
    cats = categories if categories else _arxiv_categories()
    cutoff = datetime.now(tz=UTC) - timedelta(days=_MAX_AGE_DAYS)
    out: list[ArticleCandidate] = []

    for cat in cats:
        feed_url = f"https://rss.arxiv.org/rss/{cat}"

        def parse_arxiv(url: str) -> list[ArticleCandidate]:
            local: list[ArticleCandidate] = []
            try:
                parsed = feedparser.parse(url)
            except Exception:
                logger.exception("arXiv RSS parse failed cat=%s", cat)
                return local
            for entry in getattr(parsed, "entries", []) or []:
                title = (getattr(entry, "title", None) or "").strip().replace("\n", " ")
                link = (getattr(entry, "link", None) or "").strip()
                if not title or not link:
                    continue
                pub = _parse_feed_date(entry)
                if pub < cutoff:
                    continue
                desc = getattr(entry, "description", "") or ""
                summary = (getattr(entry, "summary", None) or desc or "").strip()
                author = None
                if getattr(entry, "author", None):
                    author = str(entry.author).split(",")[0].strip()[:512] or None
                local.append(
                    ArticleCandidate(
                        title=title,
                        url=link,
                        source=f"arXiv · {cat}",
                        body=summary[:500_000],
                        author=author,
                        published_at=pub,
                        source_feed_url=feed_url,
                    )
                )
            return local

        try:
            batch = await asyncio.to_thread(parse_arxiv, feed_url)
            out.extend(batch)
        except Exception:
            logger.exception("arXiv feed failed cat=%s", cat)
    return _article_json(out)


@activity.defn
async def ingest_articles(articles: list[dict]) -> NewsIngestResult:
    settings = get_settings()
    token = (settings.platform_internal_service_token or "").strip()
    if not token:
        msg = "PLATFORM_INTERNAL_SERVICE_TOKEN is not set; cannot ingest news"
        logger.error(msg)
        raise RuntimeError(msg)

    base = settings.platform_api_base_url.rstrip("/")
    ingest_url = f"{base}/api/internal/v1/news/items"
    headers = {"Authorization": f"Bearer {token}"}
    sem = asyncio.Semaphore(10)

    models = [ArticleCandidate.model_validate(x) for x in articles]

    async def one(client: httpx.AsyncClient, art: ArticleCandidate) -> tuple[str, ...]:
        body = IngestNewsItemV1Request(
            title=art.title,
            url=art.url,
            source=art.source,
            body=art.body or "",
            author=art.author,
            published_at=art.published_at,
            source_feed_url=art.source_feed_url,
        )
        payload = body.model_dump(by_alias=True, mode="json")
        async with sem:
            try:
                r = await client.post(ingest_url, json=payload, headers=headers)
                r.raise_for_status()
                data = IngestNewsItemV1Response.model_validate(r.json())
            except Exception:
                logger.exception("ingest failed url=%s", art.url)
                return ("error",)
        if data.status == "created":
            return ("created",)
        if data.status == "duplicate":
            return ("duplicate",)
        return ("error",)

    async with httpx.AsyncClient(timeout=120.0) as client:
        rows = await asyncio.gather(*(one(client, a) for a in models))
    created = sum(1 for x in rows if x == ("created",))
    duplicates = sum(1 for x in rows if x == ("duplicate",))
    errors = sum(1 for x in rows if x == ("error",))

    logger.info(
        "ingest_articles done created=%s duplicates=%s errors=%s",
        created,
        duplicates,
        errors,
    )
    return NewsIngestResult(created=created, duplicates=duplicates, errors=errors)
