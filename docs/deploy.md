# Deploying

## On one server with Docker Compose

One machine with Docker runs everything from [`docker-compose.yml`](../docker-compose.yml): Postgres (never published), a one-shot `migrate`, the brain, the realtime worker, the three mock MCP servers (GitHub, Jira, GitLab), the board, and Caddy on ports 80 and 443. Caddy serves the board at `/` and the brain at `/api/*` (prefix stripped), answers 404 for `/api/internal/*` (the worker calls the brain directly on the compose network), and hides `/api/docs` and `/api/openapi.json` unless `EXPOSE_API_DOCS=true`.

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
