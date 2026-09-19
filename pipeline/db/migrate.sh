#!/bin/sh
# Apply every pipeline/db/migrations/*.sql not yet recorded in
# schema_migrations, in filename order, each in its own transaction.
#
# Runs as the db-migrate compose service before api starts. Safe to re-run:
# applied files are skipped. Uses psql from the postgres image, so there is
# no runtime dependency beyond the database itself.
#
# Constraints of the wrap:
# - Every file runs inside one transaction the runner opens and closes, so
#   files must not contain BEGIN/COMMIT of their own, and statements Postgres
#   refuses in a transaction block (CREATE INDEX CONCURRENTLY, ALTER TYPE ...
#   ADD VALUE) cannot be used. Add a marker convention when the first one is
#   needed; until then, plain CREATE INDEX on a live traces table locks it
#   for the duration of the build.
# - Files are fed to psql, not to the server directly, so backslash
#   meta-commands run and :var tokens outside string literals are
#   substituted. Write plain SQL only.
# - Applied files are immutable. The md5 of each file is recorded and a
#   later run refuses to start if a recorded file's content changed, so a
#   fix goes in a new file. Reverting is manual: undo the DDL by hand, then
#   DELETE FROM schema_migrations WHERE filename = '<file>'.
# - No lock between runners. Two overlapping `compose up` calls can both
#   pass the applied check; the loser fails on "already exists" and a
#   re-run fixes it. Not worth a single-session rewrite for one operator.
set -eu

: "${PGHOST:?}" "${PGUSER:?}" "${PGPASSWORD:?}" "${PGDATABASE:?}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations}"

# Bounded: an unreachable database must fail the service, not hang compose
# forever while vllm sits loaded on a billed GPU.
attempts=0
until pg_isready -q; do
  attempts=$((attempts + 1))
  if [ "$attempts" -ge 60 ]; then
    echo "postgres at $PGHOST not ready after ${attempts}s, giving up" >&2
    exit 1
  fi
  echo "waiting for postgres at $PGHOST"
  sleep 1
done

psql -v ON_ERROR_STOP=1 -q <<'SQL'
SET client_min_messages = warning;
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
SQL

applied=0
for path in "$MIGRATIONS_DIR"/*.sql; do
  # An empty directory leaves the glob unexpanded; nothing to apply.
  [ -e "$path" ] || continue
  name=$(basename "$path")
  # Read the file before the pipe. A cat failure inside the pipe would only
  # abort the producer: psql would see a lone BEGIN, exit 0, and the run
  # would report the file as applied without recording it.
  sql=$(cat "$path")
  sum=$(md5sum "$path" | cut -d' ' -f1)
  # :'name' is psql's quoted-literal substitution, so the filename never
  # lands in the SQL as raw text. It only applies to stdin, not -c.
  recorded=$(echo "SELECT checksum FROM schema_migrations WHERE filename = :'name'" \
    | psql -tAq -v ON_ERROR_STOP=1 -v name="$name")
  if [ -n "$recorded" ]; then
    if [ "$recorded" != "$sum" ]; then
      echo "$name was applied with checksum $recorded but is now $sum;" \
        "applied migrations are immutable, put the change in a new file" >&2
      exit 1
    fi
    continue
  fi
  echo "applying $name"
  # One transaction per file: a failed migration leaves nothing half-applied
  # and nothing recorded, so the fix is to correct the file and re-run.
  # printf's trailing newline keeps a final comment line in the file from
  # swallowing the INSERT.
  {
    echo "BEGIN;"
    printf '%s\n' "$sql"
    echo "INSERT INTO schema_migrations (filename, checksum) VALUES (:'name', :'sum');"
    echo "COMMIT;"
  } | psql -v ON_ERROR_STOP=1 -v name="$name" -v sum="$sum" -q
  applied=$((applied + 1))
done

echo "migrations up to date ($applied applied this run)"
