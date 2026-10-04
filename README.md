# SkyRoom

SkyRoom is a meeting app for software teams with an AI teammate, Polaris, that you call on by name during a meeting or ask about past meetings, GitHub and Jira. After a meeting it writes the report, decisions and task drafts for people to review; nothing is written to GitHub or Jira without a person's approval.

The product and agent names live in [`contracts/identity.json`](contracts/identity.json); code reads them from there.

## The parts

| Folder | What it is |
| --- | --- |
| `board/` | The web app (Next.js). Calls the brain over HTTP at `NEXT_PUBLIC_API_URL` and joins LiveKit rooms with tokens the brain issues. |
| `brain/` | The API (FastAPI): meetings, asking Polaris, the after-meeting write-up, meeting memory. Stores everything in Postgres with pgvector; reasons with Gemini (OpenRouter as a fallback); reads GitHub and Jira through MCP servers; uses ElevenLabs for voices and the report read aloud. CLI: `brain`. |
| `realtime/` | The LiveKit agent worker: transcribes each speaker with ElevenLabs, sends final transcript segments and invocations to the brain's `/internal` routes (authenticated with `BRAIN_INTERNAL_TOKEN`), and speaks an answer when a participant chooses Speak. |
| `world/` | The demo world: mock GitHub and Jira MCP servers over `mock-data/`, with a write journal (overlay) that `world-reset` clears. |
| `contracts/` | Shared request, event and data shapes, in Python and TypeScript. |
| migrations | SQL migrations for Postgres, applied by `brain migrate` (in `db/migrations` once #81 lands). |

```
browser ── /       ──> board
        ── /api/*  ──> brain ──> Postgres + pgvector
                             ──> world-github / world-jira (or real GitHub and Jira MCP servers)
                             ──> Gemini, OpenRouter, ElevenLabs, LiveKit API
        ── WebRTC  ──> LiveKit Cloud <── realtime worker ── /internal/* ──> brain
```

Settings come from one `.env` at the repo root in development ([`.env.example`](.env.example) lists them) and from `deploy/.env` on the server ([`deploy/.env.example`](deploy/.env.example)). A missing credential makes that feature unavailable; it is never faked.

## Local development

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), Node with pnpm, and Docker if you want Postgres from compose.

```sh
uv sync --all-packages
pnpm install
cp .env.example .env    # then fill in what you need
```

A local Postgres 16 with pgvector, either from compose (set `POSTGRES_PASSWORD` in `.env` first; the port is published on 127.0.0.1 only by the dev override):

```sh
docker compose -f docker-compose.yml -f deploy/compose.dev.yml up -d postgres
# DATABASE_URL=postgresql://moe:<POSTGRES_PASSWORD>@localhost:5432/moe
```

or embedded, without Docker (prints the `DATABASE_URL` to use; the server keeps running in the background):

```sh
uv run python -c "import pgserver; print(pgserver.get_server('.pgdata', cleanup_mode=None).get_uri())"
```

Then apply the migrations and run each part in its own terminal:

```sh
uv run brain migrate
uv run uvicorn brain.main:app --reload --port 8000   # the brain, http://localhost:8000/docs
uv run world-github-mcp                              # mock GitHub MCP, http://localhost:8101/mcp
uv run world-jira-mcp                                # mock Jira MCP, http://localhost:8102/mcp
pnpm --filter board dev                              # the board, http://localhost:3000
```

`uv run world-reset` clears the mocks' write journal. The realtime worker needs LiveKit and ElevenLabs credentials: `uv run realtime start` (its room entrypoint is still being built, so it does not join meetings yet).

## Tests

```sh
uv run pytest brain realtime world                  # unit and API tests; live tests are deselected
BRAIN_TEST_STORE=postgres uv run pytest brain       # the API tests on embedded Postgres instead of memory
uv run ruff check . && uv run ruff format --check .
pnpm --filter board typecheck
```

Live tests call real services and spend quota. They skip unless `.env` has what they need: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL` and `GEMINI_EMBEDDING_DIM` for most; `OPENROUTER_API_KEY` and `OPENROUTER_MODELS` for the fallback; `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` and `ELEVENLABS_TTS_MODEL` for speech. Run only the ones you need:

```sh
uv run pytest brain -m live -k <name> -rs
```

## Demo data and accounts

These commands arrive with #82 (seed) and #81 (accounts); check them against those PRs once merged.

```sh
uv run world-seed --snapshot demo                   # #82: the DropSubs team, settings, past meetings
uv run world-seed --snapshot demo --reset           # #82: remove the seeded team first
uv run brain add-team --id <id> --name <name>       # #81
uv run brain add-user --team <id> --name <name> --email <email>   # #81: prints a one-time password
```

## Deploying to a server

One machine with Docker runs everything from [`docker-compose.yml`](docker-compose.yml): Postgres (never published), a one-shot `migrate`, the brain, the realtime worker, the two mock MCP servers, the board, and Caddy on ports 80 and 443. Caddy serves the board at `/` and the brain at `/api/*` (prefix stripped), answers 404 for `/api/internal/*` (the worker calls the brain directly on the compose network), and hides `/api/docs` and `/api/openapi.json` unless `EXPOSE_API_DOCS=true`.

1. Point the domain's DNS A/AAAA record at the server and open ports 80 and 443. Caddy gets the certificate itself.
2. On the server, clone the repo and write the settings:
   ```sh
   cp deploy/.env.example deploy/.env    # then fill it in; it explains each value
   ```
3. Build and start everything. `migrate` applies new migrations before the brain starts:
   ```sh
   docker compose --env-file deploy/.env up -d --build
   docker compose --env-file deploy/.env ps
   ```
   To apply migrations by hand: `docker compose --env-file deploy/.env run --rm migrate`.
4. Load the demo and create the first account (from #82 and #81; the brain image includes the world CLIs):
   ```sh
   docker compose --env-file deploy/.env run --rm brain world-seed --snapshot demo
   docker compose --env-file deploy/.env run --rm brain brain add-team --id <id> --name <name>
   docker compose --env-file deploy/.env run --rm brain brain add-user --team <id> --name <name> --email <email>
   ```

Clear the mock servers' write journal with `docker compose --env-file deploy/.env exec world-github world-reset`. Logs: `docker compose --env-file deploy/.env logs -f brain`.

**Updating:** pull the new commit, then run `docker compose --env-file deploy/.env up -d --build` again; it rebuilds the images, runs `migrate` and restarts what changed.

**Backups:** `deploy/backup.sh` writes `backups/moe-<UTC time>.sql.gz` (a `pg_dump` from the postgres container; pass a directory to write elsewhere). Nightly from cron: `0 3 * * * cd /path/to/repo && deploy/backup.sh`. The script's header shows how to restore.
