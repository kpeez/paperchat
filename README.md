# PaperChat

PaperChat is a local web app for grounded chat over user-owned PDFs.

- `frontend/` — React + Vite app
- `backend/` — Python API (FastAPI + SQLite + sqlite-vec)

All data lives under `~/.paperchat/` (database file + model cache).

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and [pnpm](https://pnpm.io/)

## Quick Start

Launch the app from the repo root:

```bash
uv run paperchat launch
```

This command boots missing environments when needed, runs backend migrations, starts the backend and
frontend together, and opens the browser to `http://127.0.0.1:5173`.

Use `--no-open` if you want to keep the browser closed:

```bash
uv run paperchat launch --no-open
```

Set `PAPERCHAT_PORT` before launch if you need the backend on a non-default port. The launcher
will keep the frontend pointed at that backend automatically.

## Direct Commands

Keep the direct backend/frontend commands for debugging when you need to isolate one side of the
stack.

Backend:

```bash
cd backend
uv sync --group dev
uv run alembic upgrade head
uv run paperchat-backend
```

Frontend:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev --host 127.0.0.1 --port 5173
```

If you override the backend port for direct runs, keep the frontend aligned explicitly:

```bash
cd backend
PAPERCHAT_PORT=9812 uv run paperchat-backend

cd frontend
VITE_API_URL=http://127.0.0.1:9812 pnpm dev --host 127.0.0.1 --port 5173
```

`PAPERCHAT_DATABASE_URL` is SQLite-only and takes precedence over `PAPERCHAT_DB_PATH`.

## Verification

Launcher:

```bash
uv sync --group dev
uv run ruff check paperchat_cli tests
uv run ruff format --check paperchat_cli tests
uv run pytest tests/test_launcher.py
```

Backend:

```bash
cd backend
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```

Frontend:

```bash
cd frontend
pnpm lint
pnpm build
```

Runtime details are documented in [`docs/local-runtime.md`](docs/local-runtime.md).
