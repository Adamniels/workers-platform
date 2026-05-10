# workers-platform

Python Temporal worker runtime for Platform workflows.

## Scope

- Hosts workflow execution logic only.
- Does not own platform system-of-record state.
- Frontend and backend repositories are out of scope for this package.

## Structure

- `app/runtime`: worker bootstrapping, registry, Temporal client wiring, config, logging.
- `app/workflows`: workflow-owned execution modules (`news_intelligence`, `side_learning`, `memory_consolidation`).
- `app/primitives`: generic reusable helpers with no product-domain business policy.
- `app/memory`: memory access adapters and retrieval/proposal interfaces.
- `app/schemas`: typed workflow input/output contracts used inside workers.
- `tests`: unit and integration test suites.

## Boundary Rules

- Workflows must not import each other directly.
- `primitives` must remain domain-agnostic.
- Memory access goes through `app/memory/client` adapters.
- Workflow-specific judgment stays in the workflow that owns it.

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Run tests:

```bash
uv run pytest
```

3. Run lint checks:

```bash
uv run ruff check .
```

4. Start worker process:

```bash
uv run python -m app.runtime.worker.main
```

## Environment

Copy `.env.example` to `.env` and adjust values:

- `TEMPORAL_SERVER_URL` (default `localhost:7233`)
- `TEMPORAL_NAMESPACE` (default `default`)
- `TEMPORAL_TASK_QUEUE` (default `platform`)
- `LOG_LEVEL` (default `INFO`)

News ingestion (Phase 1) schedule, manual trigger, and env vars: see [phase-1-operations.md](../backend-platform/docs/news/phase-1-operations.md) in `backend-platform`.
