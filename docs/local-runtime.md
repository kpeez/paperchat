# Local Runtime

PaperChat uses SQLite for persistence. No external database is needed.

## Default Launch

Use the repo-root launcher:

```bash
uv run paperchat launch
```

The launcher:

- boots missing backend/frontend environments
- runs `alembic upgrade head`
- starts the backend on `127.0.0.1:${PAPERCHAT_PORT:-9712}`
- starts the frontend on `127.0.0.1:5173`
- opens the browser unless `--no-open` is passed

## Data Directory

All runtime data lives under `~/.paperchat/`:

```
~/.paperchat/
├── paperchat.db          # SQLite database (with sqlite-vec)
└── cache/
    ├── models/           # GGUF model files
    └── huggingface/      # HuggingFace hub cache
```

Override paths with environment variables:

| Variable | Default |
|---|---|
| `PAPERCHAT_DB_PATH` | `~/.paperchat/paperchat.db` |
| `PAPERCHAT_CACHE_DIR` | `~/.paperchat/cache` |
| `PAPERCHAT_DATABASE_URL` | `sqlite:///~/.paperchat/paperchat.db` |
| `PAPERCHAT_PORT` | `9712` |

`PAPERCHAT_DATABASE_URL` must stay SQLite-based and overrides `PAPERCHAT_DB_PATH` when both are
set.

## Direct Backend Commands

Keep the direct commands around for debugging:

```bash
cd backend
uv sync --group dev
uv run alembic upgrade head
uv run paperchat-backend
```

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev --host 127.0.0.1 --port 5173
```

If the backend is running on a non-default port, set both commands explicitly:

```bash
cd backend
PAPERCHAT_PORT=9812 uv run paperchat-backend

cd frontend
VITE_API_URL=http://127.0.0.1:9812 pnpm dev --host 127.0.0.1 --port 5173
```

The database file and parent directory are created automatically on first connection.

## Reset

To start fresh, delete the data directory:

```bash
rm -rf ~/.paperchat
```

Then re-run migrations.
