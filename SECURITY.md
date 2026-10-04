# Security

Please don't open a public issue for a security problem. Report it privately through GitHub's [private vulnerability reporting](https://github.com/Daniel-c137/stormhacks-2026-MoE/security/advisories/new) instead.

SkyRoom was built during a hackathon. If you run it yourself:

- Never commit `.env` or `deploy/.env`; only the `.example` files belong in git.
- Set a long random `AUTH_SECRET` and `BRAIN_INTERNAL_TOKEN` (`python -c "import secrets; print(secrets.token_urlsafe(48))"`).
- Give GitHub tokens read-only access, and put a spending limit on model API keys.
