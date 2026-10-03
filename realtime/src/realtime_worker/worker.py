"""LiveKit agent worker. Joins each meeting room as the agent (hidden, no video tile).

Per participant track: STT -> captions on Topic.TRANSCRIPT -> final segments to the brain.
Invocations (wake phrase, follow-up, public @mention) go to the brain; answers come back as a
shared ResponseCard and are spoken only after a participant chooses Speak.
"""

from livekit.agents import AgentServer, JobContext, cli

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    raise NotImplementedError


def main() -> None:
    cli.run_app(server)
