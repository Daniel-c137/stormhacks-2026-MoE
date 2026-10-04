#!/usr/bin/env bash
# Dumps the database from the running postgres container to a dated, gzipped file.
#   deploy/backup.sh [backup-dir]    (default: ./backups, relative to the repo root)
# Restore: gunzip -c FILE | docker compose --env-file deploy/.env exec -T postgres \
#   sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
backup_dir="${1:-$repo_root/backups}"
mkdir -p "$backup_dir"
out="$backup_dir/moe-$(date -u +%Y%m%dT%H%M%SZ).sql.gz"

cd "$repo_root"
trap 'rm -f "$out.partial"' ERR
# The container already knows its user and database; nothing secret goes on this command line.
docker compose --env-file deploy/.env exec -T postgres \
  sh -c 'pg_dump --no-owner -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  | gzip > "$out.partial"
mv "$out.partial" "$out"
echo "$out"
