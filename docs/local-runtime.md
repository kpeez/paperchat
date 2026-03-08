# Local Runtime

PaperChat PR 2 introduces a single supported local database runtime:

- Docker-managed Postgres
- `pgvector` enabled through the `pgvector/pgvector` image
- host binding on `127.0.0.1:5433`
- named Docker volume `paperchat_postgres_data`

## Start

```bash
./scripts/postgres-up.sh
```

This starts the `postgres` service from [`../compose.yaml`](../compose.yaml)
and waits for the container healthcheck to report healthy.

## Status

```bash
./scripts/postgres-status.sh
```

This shows the Compose service state and the current Docker health status.

## Stop

```bash
./scripts/postgres-down.sh
```

This stops the Postgres service but leaves the named volume in place.

## Connection Contract

- host: `127.0.0.1`
- port: `5433`
- database: `paperchat`
- user: `paperchat`
- password: `paperchat`
