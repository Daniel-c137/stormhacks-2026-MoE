# SkyRoom

SkyRoom is a meeting app for software teams with an AI teammate, Polaris, that you call on by name during a meeting or ask about past meetings, GitHub and Jira. After a meeting it writes the report, decisions and task drafts for people to review; nothing is written to GitHub or Jira without a person's approval.

The product and agent names live in [`contracts/identity.json`](contracts/identity.json); code reads them from there.

## The parts

| Folder | What it is |
| --- | --- |
| `board/` | The web app (Next.js). Calls the brain over HTTP at `NEXT_PUBLIC_API_URL` (`/api` in production) and joins LiveKit rooms with tokens the brain issues. Every screen calls the brain's real routes with the session from `POST /auth/login`. |
| `brain/` | The API (FastAPI): meetings, asking Polaris, the after-meeting write-up, meeting memory. Stores everything in Postgres with pgvector; reasons with Gemini (OpenRouter as a fallback); reads GitHub, GitLab and Jira through MCP servers (GitHub's hosted server with the team's own token once an admin connects it); uses ElevenLabs for voices and the report read aloud. CLI: `brain`. |
| `realtime/` | The LiveKit agent worker: transcribes each speaker with ElevenLabs, sends final transcript segments and invocations to the brain's `/internal` routes (authenticated with `BRAIN_INTERNAL_TOKEN`), and speaks an answer when a participant chooses Speak. Before acting on an answer card (Speak, Post in chat, Dismiss) it asks the brain whether the team's "who may allow" setting lets that participant; if the brain can't say, it refuses. |
| `world/` | The demo world: mock GitHub, GitLab and Jira MCP servers over `mock-data/`, with a write journal (overlay) that `world-reset` clears. DropSubs has two GitHub repositories (`dropsubs/dropsubs`, `dropsubs/website`) and a GitLab project (`dropsubs/infra`). |
| `contracts/` | Shared request, event and data shapes, in Python and TypeScript. |
| `db/migrations` | SQL migrations for Postgres, applied by `brain migrate`. |

```
browser ── /       ──> board
        ── /api/*  ──> brain ──> Postgres + pgvector
                             ──> world-github / world-gitlab / world-jira (or the real MCP servers)
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
uv run world-gitlab-mcp                              # mock GitLab MCP, http://localhost:8103/mcp
pnpm --filter board dev                              # the board, http://localhost:3000
```

With `NEXT_PUBLIC_API_URL=/api` (as in `.env.example`), `next dev` forwards `/api/*` to `BRAIN_URL`, so the board and the brain share an origin locally just as behind Caddy; the brain sends no CORS headers. Meetings need LiveKit: for a local server, `docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev --bind 0.0.0.0 --node-ip 127.0.0.1` with `LIVEKIT_URL=ws://localhost:7880`, `LIVEKIT_API_KEY=devkey` and `LIVEKIT_API_SECRET=secret` (`--node-ip` makes it advertise an address the browser can reach from outside the container).

`uv run world-reset` clears the mocks' write journal.

The realtime worker joins every meeting as the agent: `uv run realtime start` (or `dev` to reload on changes). It needs `LIVEKIT_*`, `ELEVENLABS_API_KEY`, `ELEVENLABS_STT_MODEL` (e.g. `scribe_v2_realtime`), `ELEVENLABS_TTS_MODEL`, `ELEVENLABS_VOICE_ID`, `BRAIN_URL` and `BRAIN_INTERNAL_TOKEN`, and exits naming whatever is missing. When the host turns on live translation for a meeting (off by default, set before anyone joins), speech in another language is shown and saved in English: Scribe detects each utterance's language and the brain translates it with its LLM. Without it, Scribe is pinned to `ELEVENLABS_STT_LANGUAGE` (default `en`). Translation runs on `TRANSLATION_MODEL` (default `GEMINI_MODEL`) with one attempt and no paid fallback; if it fails, captions show the words as said. Wake detection reads the English, so "Polaris, …" works in any language once translated; a question left untranslated (the model down) won't wake Polaris. Polaris shows it is listening as soon as a live caption starts with its name, while you are still talking; only the finished sentence asks the question. Scribe ends a sentence after `ELEVENLABS_VAD_SILENCE_SECONDS` of silence (default 1.0), and Polaris starts working then. LiveKit dispatches it automatically to each room created while it is registered, so start it before people join; it leaves any room that is not a live meeting.

## Tests

```sh
uv run pytest brain realtime world                  # unit and API tests; live tests are deselected
BRAIN_TEST_STORE=postgres uv run pytest brain       # the API tests on embedded Postgres instead of memory
uv run ruff check . && uv run ruff format --check .
pnpm --filter board typecheck
pnpm --filter board test                            # the board's link and next-path helpers, with Node's test runner
```

Live tests call real services and spend quota. They skip unless `.env` has what they need: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL` and `GEMINI_EMBEDDING_DIM` for most; `OPENROUTER_API_KEY` and `OPENROUTER_MODELS` for the fallback (`OPENROUTER_EMBEDDING_MODEL` for the embeddings fallback); `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` and `ELEVENLABS_TTS_MODEL` for speech. Run only the ones you need:

```sh
uv run pytest brain -m live -k <name> -rs
```

The worker's end-to-end test (`realtime/tests/test_live_join.py`) runs the brain on embedded Postgres, the worker and a synthetic participant who speaks a macOS `say` recording, against a local LiveKit server. It spends about 6 s of Scribe audio, two Gemini answers and one short spoken answer:

```sh
LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
ELEVENLABS_STT_MODEL=scribe_v2_realtime uv run pytest realtime -m live -k join -s
```

## Teams, accounts and demo data

There is no open sign-up: only someone an admin invited can create an account. Create a team and its first admin with the brain's CLI; `add-user` prints a one-time password unless you pipe one in with `--password-stdin`:

```sh
uv run brain add-team --id <team-id> --name "<team name>"
uv run brain add-user --team <team-id> --name "<full name>" --email <email> --admin
```

Only an admin changes the team's settings (connectors, the agent's voice and fact-checking, who may allow answers, time zone) and invites the other people, in Settings → Members → Invite (`POST /team/accounts` with `invite`). The admin enters only the email (and the role); the person shows under a name made from it until they open the copied link (`/login?mode=signup`) and create their account with that email, choosing their name and password, or with Google (which gives their name), and join the team. They show as Invited until then, and can already be picked as a meeting invitee or a task owner. The sign-up page doesn't check the mailbox (the brain sends no email), so whoever first creates the account for an invited email gets it: invite people shortly before you send them the link. An email another team has already invited can't be invited or given a password (409); removing a member isn't built yet. The API can still create a login with a generated password (`POST /team/accounts` without `invite`, with a name), as `brain add-user` does; the board only invites. What a meeting's host does (ending it, changing its invitees, retrying its write-up) an admin may do too, so a meeting whose host left can still be managed. Approving a meeting's push to Jira is an admin's alone. Everyone keeps their own profile, photo and password. `add-user` without `--admin` adds someone who is not an admin, and leaves an existing admin one. To grant or revoke admin later (the brain checks on every request, so it takes effect at once; a team's last admin can't be revoked):

```sh
uv run brain set-admin --email <email>
uv run brain set-admin --email <email> --revoke
```

Google sign-in is on when `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and `GOOGLE_REDIRECT_URL` are all set. It signs in the account with the Google account's email, or takes that email's invite, but only for a Gmail address or a Google Workspace account: Google can't vouch for any other address on a Google account, so those people use a password. In production `GOOGLE_REDIRECT_URL` is `https://<domain>/api/auth/google/callback`, and it must match the redirect URI registered for the OAuth client in Google Cloud; locally it's `http://localhost:3000/api/auth/google/callback`, through the board's `/api` rewrite. The brain can't work this URL out itself behind the proxy, and an `https` URL makes the sign-in cookie Secure. The brain hands a Google sign-in to the board with a one-time code that works once, within a minute, and only in the browser that signed in (an HttpOnly cookie holds its other half). The codes are kept in the brain's memory, so run a single brain process.

The demo seed loads the DropSubs team, its settings and its past meetings (written up by the real models) and gives each person a login with their seeded email; `danial@dropsubs.example` is the team's admin. `WORLD_SEED_PASSWORD` sets one demo password for everyone; otherwise each person gets a generated one, printed once. Running it again skips what's already there.

```sh
uv run world-seed --snapshot demo                   # the DropSubs team, logins, settings and past meetings
uv run world-seed --snapshot demo --reset           # remove the seeded team first (logins are kept)
```

The seed connects DropSubs to `dropsubs/dropsubs` and `dropsubs/website` on GitHub, `dropsubs/infra` on GitLab and the `DS` Jira project. It writes settings only for a new team, so a team seeded earlier keeps its connectors: add the new ones in Settings, or seed again with `--reset`.

On Gemini's free tier, set `WORLD_SEED_EMBEDS_PER_MINUTE=90`. The free tier also allows only 1,000 embedded texts a day. Each past meeting embeds about 35 windows of its transcript plus its summary, decisions and tasks, so the four meetings fit in one day; if a run is stopped by the quota anyway, a re-run the next day carries on where the last one stopped.

## Connectors

A team admin connects GitHub repositories and GitLab projects (up to 10 of each, each at an optional branch or tag) and the Jira site and project in Settings, under Connectors (`PUT /settings/connectors`). Polaris reads every connected repository when it answers, searches code and checks facts, unless a question names one; sources name their repository (`dropsubs/website#7`, `dropsubs/infra!4` for a GitLab merge request). Each connector shows Connected, Not set up or Not reachable, checked live against its MCP server (`GET /settings/connectors`).

### Reading the team's real GitHub

Without anything more, the team's repositories are read through `GITHUB_MCP_URL` with no credentials: the demo's mock GitHub. To have Polaris read the team's real issues, pull requests (with their checks and reviews) and code, an admin connects GitHub: in Settings, under Connectors, choose Add connector, then "Connect GitHub", and paste a [fine-grained personal access token](https://github.com/settings/personal-access-tokens/new) (`PUT /settings/github/account`). The token needs only read access, to the team's repositories: Metadata, Contents, Issues, Pull requests and Commit statuses (classic `ghp_` tokens are refused). The brain asks GitHub's REST API (`GITHUB_API_URL`) whose token it is and checks that it reads each connected repository, naming any repository or permission it lacks; it stores the token encrypted with a key derived from `AUTH_SECRET` and never sends it to a browser. The row then shows `@login`, and a repository added later must be readable with the token too.

From then on every GitHub read for that team (answers, fact checks, code search, the connector status) goes to GitHub's hosted MCP server (`GITHUB_HOSTED_MCP_URL`, `https://api.githubcopilot.com/mcp/`) with the token as a bearer header, asking only for read-only tools; it never falls back to the mock. The hosted server is available to every GitHub account and needs no Copilot subscription. A token GitHub no longer accepts shows the GitHub connector as Not reachable, with the reason; Disconnect forgets the token and the team reads `GITHUB_MCP_URL` again. Teams without a token are unaffected.

### Pushing tasks to Jira

To have approved task drafts created as real Jira issues, an admin connects the team's Jira Cloud account: in Settings, under Connectors, choose Add connector, then "Jira account, to push tasks", and enter the site (`your-team.atlassian.net`), the project key, an Atlassian account's email and an [API token](https://id.atlassian.com/manage-profile/security/api-tokens) for it (a plain token, not one "with scopes") (`PUT /settings/jira/account`). The brain checks the account and the project with Jira before saving, calls only `https://<name>.atlassian.net`, stores the token encrypted with a key derived from `AUTH_SECRET` and never sends it to a browser. Disconnect forgets it.

The account has its own site and project, shown as its own row. Jira is asked about it when it is connected and at every push, not while Settings is open, so a token that has since expired shows there as connected until the next push says otherwise. The Jira project row above it is what Polaris reads (answers, agenda suggestions, fact checks) through `JIRA_MCP_URL`, the mock in the demo; connecting or disconnecting the account does not change it, so the demo's Jira answers and its earlier tasks stay as they are.

On a meeting's report, an admin then presses Push: each included draft becomes a Task in the account's project, as the connected account, with the meeting, the quoted moment and the approver in its description. The owner is assigned when Jira has exactly one assignable account with that person's email (or name); otherwise the issue is created unassigned and the page says so. A due date the project's create screen does not take is left out the same way. Each pushed task keeps the link to its issue. One push of a meeting runs at a time and each key is saved as soon as Jira has made the issue, so a retry after a failure creates only what is still missing. If the account's project has the same key as the project Polaris reads, the same key (say `DS-12`) can name two different issues: the real one a task links to, and the mock's that Polaris answers about. Without a connected account the board does not offer the push; the API then falls back to the Jira MCP server (`JIRA_MCP_URL`), as the `brain push` CLI does.

## Deploying to a server

One machine with Docker runs everything from [`docker-compose.yml`](docker-compose.yml): Postgres (never published), a one-shot `migrate`, the brain, the realtime worker, the three mock MCP servers (GitHub, Jira, GitLab), the board, and Caddy on ports 80 and 443. Caddy serves the board at `/` and the brain at `/api/*` (prefix stripped), answers 404 for `/api/internal/*` (the worker calls the brain directly on the compose network), and hides `/api/docs` and `/api/openapi.json` unless `EXPOSE_API_DOCS=true`.

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
4. Create the first team and its admin (`add-user` prints a one-time password); the admin then creates the other accounts in Settings:
   ```sh
   docker compose --env-file deploy/.env exec brain brain add-team --id <team-id> --name "<team name>"
   docker compose --env-file deploy/.env exec brain brain add-user --team <team-id> --name "<full name>" --email <email> --admin
   ```
   Or load the demo instead; it creates the DropSubs people with logins (the brain image includes the world CLIs):
   ```sh
   docker compose --env-file deploy/.env exec brain world-seed --snapshot demo
   ```

Clear the mock servers' write journal with `docker compose --env-file deploy/.env exec world-github world-reset`. Logs: `docker compose --env-file deploy/.env logs -f brain`.

**Updating:** pull the new commit, then run `docker compose --env-file deploy/.env up -d --build` again; it rebuilds the images, runs `migrate` and restarts what changed.

**Backups:** `deploy/backup.sh` writes `backups/moe-<UTC time>.sql.gz` (a `pg_dump` from the postgres container; pass a directory to write elsewhere). Nightly from cron: `0 3 * * * cd /path/to/repo && deploy/backup.sh`. The script's header shows how to restore.
