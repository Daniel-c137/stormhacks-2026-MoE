# SkyRoom

**A meeting app for software teams, with an AI teammate that remembers.**

Built at [StormHacks 2026](https://stormhacks.com). Try it at **[skyroomapp.tech](https://skyroomapp.tech)**.

Software teams lose most of what gets said in meetings. "Didn't we decide this last sprint?" goes unanswered, someone says a fix shipped while the PR is still open, and action items get copied into Jira by hand afterwards. SkyRoom is a video meeting app with an agent, **Polaris**, built in. Polaris knows your past meetings, your GitHub and your Jira, and it acts only when someone asks it to.

## What it does

**During a meeting**

- **Live captions per speaker.** Each participant's audio is transcribed on its own stream, so every line has the right name and timestamp.
- **Ask by voice or chat.** Say "Polaris, what did we decide about the billing retry?" or type `@Polaris`. Answers cite their sources: `meeting 3 · 14:02`, `dropsubs/website#7`, `DS-12`.
- **No surprise speech.** A voice question produces an answer card everyone sees. Polaris speaks only when someone presses **Speak**.
- **Private stays private.** A private question gets a private answer, and private chat is never saved, indexed or summarized.
- **Private fact checks.** If someone says something the team's records contradict, Polaris tells only that person, with sources.
- **Agenda tracking, catch-up for late joiners, and optional live translation** into English.

**After a meeting**

- A report with the overview, decisions (each linked to the moment it was made), open questions and blockers.
- **Task drafts** with owners and due dates only when someone actually said them.
- **Push to Jira after approval.** Nothing is written to GitHub or Jira without a person approving it.
- Every meeting goes into team memory for the next one to ask about.

## How it's built

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

## Quick start

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), Node with [pnpm](https://pnpm.io), a Postgres 16 with pgvector (Docker, or the embedded one in [docs/development.md](docs/development.md)), a [LiveKit](https://livekit.io) server, and API keys for [Gemini](https://ai.google.dev) and [ElevenLabs](https://elevenlabs.io).

```sh
uv sync --all-packages
pnpm install
cp .env.example .env                  # fill in DATABASE_URL, AUTH_SECRET, model and LiveKit keys
uv run brain migrate
uv run world-seed --snapshot demo     # optional: the DropSubs demo team, people and past meetings
```

Then run the brain, the mock MCP servers, the realtime worker and the board, each in its own terminal; [docs/development.md](docs/development.md) has the exact commands. The demo world uses mock GitHub, GitLab and Jira servers over a fictional company, DropSubs, so you can try everything without connecting your own accounts.

## Documentation

| Guide | Covers |
| --- | --- |
| [Development](docs/development.md) | Local setup, Postgres, LiveKit, the realtime worker, tests and live tests |
| [Teams, accounts and demo data](docs/accounts.md) | Creating teams and admins, invites, Google sign-in, the demo seed |
| [Connectors](docs/connectors.md) | GitHub, GitLab and Jira: reading real repositories, pushing approved tasks |
| [Deploying](docs/deploy.md) | One server with Docker Compose and Caddy, updates and backups |

## Team

Built by [Danial](https://github.com/Daniel-c137), [Hossein Zaredar](https://github.com/HosseinZaredar), [Mohammad Reza Sadeghian](https://github.com/sadeghianmr) and [Reyhaneh Ahani](https://github.com/ReyhanehAhani).

## License

[MIT](LICENSE). To report a security issue, see [SECURITY.md](SECURITY.md).
