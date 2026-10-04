# Development

Running SkyRoom on your machine, and its tests. Commands run from the repo root; settings come from `.env` ([`.env.example`](../.env.example) explains each one).

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

The realtime worker joins every meeting as the agent: `uv run realtime start` (or `dev` to reload on changes). It needs `LIVEKIT_*`, `ELEVENLABS_API_KEY`, `ELEVENLABS_STT_MODEL` (e.g. `scribe_v2_realtime`), `ELEVENLABS_TTS_MODEL`, `ELEVENLABS_VOICE_ID`, `BRAIN_URL` and `BRAIN_INTERNAL_TOKEN`, and exits naming whatever is missing. When the host turns on live translation for a meeting (off by default, set before anyone joins), speech in another language is shown and saved in English: Scribe detects each utterance's language and the brain translates it with its LLM. Without it, Scribe is pinned to `ELEVENLABS_STT_LANGUAGE` (default `en`). Translation runs on `TRANSLATION_MODEL` (default `GEMINI_MODEL`) with one attempt and no paid fallback; if it fails, captions show the words as said. Wake detection reads the English, so "Polaris, …" works in any language once translated; a question left untranslated (the model down) won't wake Polaris. Polaris shows it is listening as soon as a live caption starts with its name, while you are still talking; only the finished sentence asks the question. Scribe ends a sentence after `ELEVENLABS_VAD_SILENCE_SECONDS` of silence (default 1.0), and Polaris starts working then. The worker asks the brain to keep time against the agenda every `AGENDA_TICK_SECONDS` (default 10), and Gemini labels what was said in batches. With `JEV_MODEL` set (e.g. `typesafe/jev-1.13`, called through OpenRouter with `OPENROUTER_API_KEY`), Jev does the labelling instead and the worker also checks about 3.5 s after every caption, so a finished item is ticked within a few seconds of its last sentence; the brain and the worker both read `JEV_MODEL` and `OPENROUTER_API_KEY`. With live translation on, checks wait 4 s more for translated captions. LiveKit dispatches it automatically to each room created while it is registered, so start it before people join; it leaves any room that is not a live meeting.

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

