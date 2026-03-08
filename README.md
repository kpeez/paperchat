# PaperChat

PaperChat is a local web app for grounded chat over user-owned PDFs.

The live code paths are:

- `frontend/` for the React app
- `backend/` for the local Python API
- `compose.yaml` for the local Postgres + `pgvector` runtime

## Local Development

Start the database:

```bash
./scripts/postgres-up.sh
```

Start the backend:

```bash
cd backend
uv sync --locked --group dev
uv run alembic upgrade head
uv run paperchat-backend
```

Start the frontend in a second shell:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

## Verification

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
