-- Least-privilege role for the api's trace writer.
--
-- The api is the only service in the serving path, so it is the one most
-- exposed to a bug or a bad request. It needs exactly one capability: add a
-- row to traces. Everything else the pipeline does to that table — scoring,
-- curation status, export, deletion — belongs to jobs that run outside the
-- request path and connect as their own roles.
--
-- What this role cannot do, deliberately:
--   SELECT   cannot read back other users' conversations
--   UPDATE   cannot overwrite a judge score or a curation decision
--   DELETE   cannot remove training signal
--   DDL      cannot drop the table the training set is built from
--
-- The password comes from API_DB_PASSWORD in .env, substituted by migrate.sh.
-- Roles are cluster-wide, not per-database, so this survives a database drop
-- but not a volume drop (`docker compose down -v`), which is also what resets
-- schema_migrations — the two stay consistent.

CREATE ROLE api_writer LOGIN PASSWORD :'api_db_password';

-- CONNECT is not implicit for a new role only because PUBLIC holds it by
-- default; granting explicitly keeps this correct if PUBLIC is ever revoked.
GRANT CONNECT ON DATABASE traces TO api_writer;
GRANT USAGE ON SCHEMA public TO api_writer;

-- The whole grant. No SELECT: the writer never reads, not even its own rows.
--
-- That has one non-obvious consequence for the writer's statement. An upsert
-- guard may not name its conflict target here: `ON CONFLICT (request_id)`
-- requires SELECT on request_id and fails with "permission denied for table
-- traces" without it. The api uses the untargeted `ON CONFLICT DO NOTHING`,
-- which needs INSERT alone. Grant SELECT here and that constraint disappears —
-- along with the guarantee that the serving path cannot read conversations
-- back.
GRANT INSERT ON TABLE traces TO api_writer;
