"""Temporal activities: fetch sources and ingest into Platform API."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import socket
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
import trafilatura
from temporalio import activity

from app.runtime.config import get_settings
from app.workflows.news_intelligence.contracts import (
    ArticleCandidate,
    IngestNewsItemV1Request,
    IngestNewsItemV1Response,
    NewsEmbedResult,
    NewsIngestResult,
)

logger = logging.getLogger(__name__)

# TODO: Dont want to have any default, everything should be configured in the frontend.
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

_MAX_AGE_DAYS = 7              # discard articles older than this (all sources)
_MIN_BODY_CHARS = 1000         # articles with less body text are dropped (not worth embedding)
_MAX_ARTICLES_PER_FEED = 20    # cap entries per feed before enrichment
_FEEDPARSER_TIMEOUT_SECS = 15  # socket timeout for feedparser.parse()
_TRAFILATURA_TIMEOUT_SECS = 10 # per-article asyncio timeout for trafilatura fetch
_ENRICH_CONCURRENCY = 6        # max concurrent trafilatura fetches at once


@dataclasses.dataclass
class _FeedEntry:
    """Raw metadata from any source before body enrichment."""
    title: str
    url: str
    source: str
    author: str | None
    published_at: datetime
    source_feed_url: str | None


def _parse_feed_metadata(url: str) -> list[_FeedEntry]:
    """Parse a single RSS/Atom feed and return entry metadata only.

    Runs synchronously inside asyncio.to_thread. Sets a socket timeout so a
    hung server cannot block the thread indefinitely. Does NOT fetch article
    bodies — that happens in the async enrichment phase.
    """
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_FEEDPARSER_TIMEOUT_SECS)
    try:
        parsed = feedparser.parse(url)
    except Exception:
        logger.exception("RSS parse failed url=%s", url)
        return []
    finally:
        socket.setdefaulttimeout(old_timeout)

    feed_title = (getattr(parsed.feed, "title", None) or url)[:256]
    cutoff = datetime.now(tz=UTC) - timedelta(days=_MAX_AGE_DAYS)
    entries: list[_FeedEntry] = []

    for entry in (getattr(parsed, "entries", []) or [])[:_MAX_ARTICLES_PER_FEED]:
        title = (getattr(entry, "title", None) or "").strip()
        link = (getattr(entry, "link", None) or "").strip()
        if not title or not link:
            continue

        pub = _parse_feed_date(entry)
        if pub < cutoff:
            continue

        raw_author = getattr(entry, "author", None)
        author = str(raw_author).strip() or None if raw_author else None

        entries.append(_FeedEntry(
            title=title,
            url=link,
            source=feed_title,
            author=author,
            published_at=pub,
            source_feed_url=url,
        ))

    return entries


def _fetch_article_body(url: str) -> str:
    """Fetch and extract clean article text via trafilatura. Returns "" on failure.

    Runs synchronously inside asyncio.to_thread. Sets a socket timeout slightly
    under _TRAFILATURA_TIMEOUT_SECS so trafilatura's internal retries give up on
    their own before the asyncio cancellation fires, avoiding lingering threads.
    """
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(max(3.0, _TRAFILATURA_TIMEOUT_SECS - 2))
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
    finally:
        socket.setdefaulttimeout(old_timeout)


async def _enrich_candidates(candidates: list[_FeedEntry]) -> list[ArticleCandidate]:
    """Fetch full article body for each candidate via trafilatura, concurrently.

    Shared by all source activities. Candidates that time out or return less
    than _MIN_BODY_CHARS of clean text are dropped — short text is not worth
    embedding and would pollute the ranking layer.
    """
    sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)

    async def enrich(entry: _FeedEntry) -> ArticleCandidate | None:
        async with sem:
            try:
                body = await asyncio.wait_for(
                    asyncio.to_thread(_fetch_article_body, entry.url),
                    timeout=_TRAFILATURA_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.warning("trafilatura timeout url=%s", entry.url)
                return None

        if len(body) < _MIN_BODY_CHARS:
            logger.debug("drop short body url=%s chars=%d", entry.url, len(body))
            return None

        return ArticleCandidate(
            title=entry.title,
            url=entry.url,
            source=entry.source,
            body=body[:500_000],
            author=entry.author,
            published_at=entry.published_at,
            source_feed_url=entry.source_feed_url,
        )

    enriched = await asyncio.gather(*[enrich(c) for c in candidates])
    return [a for a in enriched if a is not None]


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

    # Phase 1: parse all feeds concurrently — feedparser only, no per-article HTTP.
    feed_results = await asyncio.gather(
        *[asyncio.to_thread(_parse_feed_metadata, u) for u in urls],
        return_exceptions=True,
    )

    candidates: list[_FeedEntry] = []
    for url, result in zip(urls, feed_results):
        if isinstance(result, BaseException):
            logger.error("RSS feed failed url=%s error=%s", url, result)
            continue
        logger.info("RSS feed parsed url=%s entries=%d", url, len(result))
        candidates.extend(result)

    if not candidates:
        logger.warning("RSS: no candidates after feed parsing")
        return []

    # Phase 2: fetch full article body for all candidates concurrently.
    # Trafilatura is always the primary source — RSS excerpts are too short to embed.
    out = await _enrich_candidates(candidates)

    logger.info(
        "RSS fetch complete candidates=%d accepted=%d dropped=%d",
        len(candidates),
        len(out),
        len(candidates) - len(out),
    )
    return _article_json(out)


@activity.defn
async def fetch_hacker_news_articles(min_score: int = 50, max_results: int = 30) -> list[dict]:
    # search_by_date returns recent stories sorted by date rather than all-time popularity.
    # We also pass the cutoff timestamp directly to Algolia so it only returns stories
    # within our age window — without this the age filter drops everything.
    cutoff = datetime.now(tz=UTC) - timedelta(days=_MAX_AGE_DAYS)
    cutoff_ts = int(cutoff.timestamp())
    api_url = (
        "https://hn.algolia.com/api/v1/search_by_date"
        f"?tags=story"
        f"&numericFilters=points%3E{min_score},created_at_i%3E{cutoff_ts}"
        f"&hitsPerPage={max_results}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(api_url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        logger.exception("Hacker News API request failed")
        return []

    candidates: list[_FeedEntry] = []

    for hit in data.get("hits", []) or []:
        title = (hit.get("title") or "").strip()
        story_url = (hit.get("url") or "").strip()
        if not title or not story_url:
            continue
        # Skip job posts — they have no editorial value as news
        if "job" in (hit.get("_tags") or []):
            continue
        created = hit.get("created_at")
        try:
            pub = datetime.fromisoformat(str(created).replace("Z", "+00:00")) if created else datetime.now(tz=UTC)
        except (TypeError, ValueError):
            pub = datetime.now(tz=UTC)
        candidates.append(_FeedEntry(
            title=title,
            url=story_url,
            source="Hacker News",
            author=None,
            published_at=pub,
            source_feed_url=None,
        ))

    if not candidates:
        logger.warning("Hacker News: no candidates after filtering")
        return []

    out = await _enrich_candidates(candidates)
    logger.info(
        "HN fetch complete candidates=%d accepted=%d dropped=%d",
        len(candidates),
        len(out),
        len(candidates) - len(out),
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
    candidates: list[_FeedEntry] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for topic in topic_list:
            try:
                await asyncio.sleep(0.5)
                r = await client.get(
                    "https://gnews.io/api/v4/search",
                    params={"q": topic, "max": str(max_per_topic), "token": key, "lang": "en"},
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
                src = art.get("source") if isinstance(art.get("source"), dict) else {}
                src_name = (src.get("name") or "GNews")[:256]
                pub_raw = art.get("publishedAt")
                try:
                    pub = datetime.fromisoformat(str(pub_raw).replace("Z", "+00:00")) if pub_raw else datetime.now(tz=UTC)
                except (TypeError, ValueError):
                    pub = datetime.now(tz=UTC)
                candidates.append(_FeedEntry(
                    title=t,
                    url=u,
                    source=src_name,
                    author=None,
                    published_at=pub,
                    source_feed_url=None,
                ))

    if not candidates:
        logger.warning("GNews: no candidates after API fetch")
        return []

    out = await _enrich_candidates(candidates)
    logger.info(
        "GNews fetch complete candidates=%d accepted=%d dropped=%d",
        len(candidates),
        len(out),
        len(candidates) - len(out),
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
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(_FEEDPARSER_TIMEOUT_SECS)
            try:
                parsed = feedparser.parse(url)
            except Exception:
                logger.exception("arXiv RSS parse failed cat=%s", cat)
                return local
            finally:
                socket.setdefaulttimeout(old_timeout)
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

    async def one(client: httpx.AsyncClient, art: ArticleCandidate) -> tuple[str, str | None]:
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
                return ("error", None)
        if data.status == "created":
            return ("created", data.id)
        if data.status == "duplicate":
            return ("duplicate", None)
        return ("error", None)

    async with httpx.AsyncClient(timeout=120.0) as client:
        rows = await asyncio.gather(*(one(client, a) for a in models))
    created_ids = [row[1] for row in rows if row[0] == "created" and row[1] is not None]
    created = len(created_ids)
    duplicates = sum(1 for x in rows if x[0] == "duplicate")
    errors = sum(1 for x in rows if x[0] == "error")

    logger.info(
        "ingest_articles done created=%s duplicates=%s errors=%s",
        created,
        duplicates,
        errors,
    )
    return NewsIngestResult(created=created, duplicates=duplicates, errors=errors, created_ids=created_ids)


@activity.defn
async def embed_news_articles(article_ids: list[str]) -> dict:
    """Embed newly created articles by calling the backend embedding endpoint.

    Accepts only the IDs that were *created* this run — duplicates already have
    embeddings and are intentionally excluded. Concurrency capped at 5 so the
    OpenAI API is not hammered from many parallel requests.
    """
    if not article_ids:
        result = NewsEmbedResult()
        logger.info("embed_news_articles skipped — no new articles")
        return result.model_dump()

    settings = get_settings()
    base = settings.platform_api_base_url.rstrip("/")
    token = (settings.platform_internal_service_token or "").strip()
    headers = {"Authorization": f"Bearer {token}"}
    sem = asyncio.Semaphore(5)
    embedded = skipped = errors = 0

    async def embed_one(client: httpx.AsyncClient, article_id: str) -> str:
        async with sem:
            try:
                r = await client.post(
                    f"{base}/api/internal/v1/news/items/{article_id}/embed",
                    headers=headers,
                )
                r.raise_for_status()
                return r.json().get("status", "error")
            except Exception:
                logger.exception("embed failed id=%s", article_id)
                return "error"

    async with httpx.AsyncClient(timeout=30.0) as client:
        statuses = await asyncio.gather(*(embed_one(client, aid) for aid in article_ids))

    for s in statuses:
        if s == "embedded":
            embedded += 1
        elif s == "skipped":
            skipped += 1
        else:
            errors += 1

    logger.info(
        "embed_news_articles done embedded=%s skipped=%s errors=%s",
        embedded,
        skipped,
        errors,
    )
    result = NewsEmbedResult(embedded=embedded, skipped=skipped, errors=errors)
    return result.model_dump()


@activity.defn
async def ensure_user_news_profile(user_id: int) -> dict:
    """Seed the user's news interest profile if it does not yet exist.

    Idempotent — safe to call on every workflow run. The backend checks
    for an existing profile and returns "exists" without touching the DB
    when one is already in place.
    """
    settings = get_settings()
    base = settings.platform_api_base_url.rstrip("/")
    token = (settings.platform_internal_service_token or "").strip()
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(
                f"{base}/api/internal/v1/news/profile/seed",
                json={"userId": user_id},
                headers=headers,
            )
            r.raise_for_status()
            status = r.json().get("status", "error")
    except Exception:
        logger.exception("ensure_user_news_profile failed user_id=%s", user_id)
        status = "error"

    logger.info("ensure_user_news_profile user_id=%s status=%s", user_id, status)
    return {"status": status}
