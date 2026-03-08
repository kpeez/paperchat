#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to stop the PaperChat local Postgres runtime." >&2
  exit 1
fi

docker compose -f "$repo_root/compose.yaml" stop postgres
