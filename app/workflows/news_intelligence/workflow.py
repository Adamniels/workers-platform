"""News intelligence ingestion workflow."""

from __future__ import annotations

import json
from datetime import timedelta

from temporalio import workflow

from app.schemas.workflow_contracts import WorkflowRunRequest, WorkflowRunResult
from app.workflows.news_intelligence.contracts import NewsIngestResult


@workflow.defn
class NewsIntelligenceWorkflow:
    """Fetch from configured sources, dedupe by URL, ingest via Platform internal API."""

    @workflow.run
    async def run(self, payload: str) -> WorkflowRunResult:
        request = WorkflowRunRequest.model_validate_json(payload)

        # Activities read their own settings; the workflow just fans out unconditionally.
        # GNews gracefully skips itself when GNEWS_API_KEY is absent.
        # RSS and arXiv respect their JSON env vars inside their own activities.
        results = await workflow.asyncio.gather(
            workflow.execute_activity(
                "fetch_rss_articles",
                None,
                start_to_close_timeout=timedelta(minutes=10),
            ),
            workflow.execute_activity(
                "fetch_hacker_news_articles",
                args=[100, 30],
                start_to_close_timeout=timedelta(seconds=30),
            ),
            workflow.execute_activity(
                "fetch_gnews_articles",
                args=[None, 5],
                start_to_close_timeout=timedelta(seconds=60),
            ),
            workflow.execute_activity(
                "fetch_arxiv_articles",
                None,
                start_to_close_timeout=timedelta(seconds=60),
            ),
            return_exceptions=True,
        )

        seen: set[str] = set()
        unique: list[dict] = []
        for batch in results:
            if isinstance(batch, BaseException):
                continue
            for item in batch:
                if not isinstance(item, dict):
                    continue
                u = (item.get("url") or "").strip().lower()
                if not u or u in seen:
                    continue
                seen.add(u)
                unique.append(item)

        ingest_raw = await workflow.execute_activity(
            "ingest_articles",
            unique,
            start_to_close_timeout=timedelta(seconds=120),
        )
        summary = NewsIngestResult.model_validate(ingest_raw)

        # Phase 2: embed only the articles created this run, then ensure the user
        # interest profile exists so the feed can be ranked by cosine similarity.
        if summary.created_ids:
            await workflow.execute_activity(
                "embed_news_articles",
                summary.created_ids,
                start_to_close_timeout=timedelta(minutes=2),
            )

        await workflow.execute_activity(
            "ensure_user_news_profile",
            1,  # primary user — single-user system for now
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Phase 3: update the long-term and short-term behavioral profiles from recent interactions.
        await workflow.execute_activity(
            "update_user_news_profile",
            1,  # primary user — single-user system for now
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Phase 4: refresh the active context embedding from declared interests and projects.
        await workflow.execute_activity(
            "update_user_news_active_context",
            1,  # primary user — single-user system for now
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Phase 5: LLM re-ranking with per-article explanations.
        await workflow.execute_activity(
            "rank_news_feed_with_llm",
            1,  # primary user — single-user system for now
            start_to_close_timeout=timedelta(seconds=120),
        )

        artifact = json.dumps(summary.model_dump(by_alias=True, mode="json"))

        return WorkflowRunResult(
            workflow_type=request.workflow_type,
            workflow_run_id=request.workflow_run_id,
            status="completed",
            artifact_refs=[artifact],
        )
