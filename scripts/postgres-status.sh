#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="$repo_root/compose.yaml"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to inspect the PaperChat local Postgres runtime." >&2
  exit 1
fi

docker compose -f "$compose_file" ps postgres

container_id="$(docker compose -f "$compose_file" ps -q postgres)"
if [[ -n "$container_id" ]]; then
  docker inspect --format 'health={{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id"
fi
