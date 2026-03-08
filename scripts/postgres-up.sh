#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$repo_root/compose.yaml"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to start the PaperChat local Postgres runtime." >&2
  exit 1
fi

docker compose -f "$compose_file" up -d postgres

container_id="$(docker compose -f "$compose_file" ps -q postgres)"
if [[ -z "$container_id" ]]; then
  echo "postgres container was not created." >&2
  exit 1
fi

for _ in {1..30}; do
  status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
  if [[ "$status" == "healthy" ]]; then
    echo "postgres is healthy on 127.0.0.1:5433"
    exit 0
  fi
  sleep 1
done

echo "postgres did not become healthy within 30 seconds." >&2
exit 1
