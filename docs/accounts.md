# Teams, accounts and demo data

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

