"""The mock MCP servers listen on localhost unless WORLD_MCP_HOST says otherwise, so a container
can serve them to its neighbours."""

import pytest

from world import github_mcp, jira_mcp
from world.config import Settings


def test_the_servers_listen_on_localhost_by_default():
    assert Settings(_env_file=None).world_mcp_host == "127.0.0.1"


@pytest.mark.parametrize(
    ("module", "port_var", "port"),
    [(github_mcp, "WORLD_GITHUB_MCP_PORT", 9101), (jira_mcp, "WORLD_JIRA_MCP_PORT", 9102)],
)
def test_main_listens_on_the_configured_host_and_port(monkeypatch, module, port_var, port):
    calls = []
    monkeypatch.setattr(module.server, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("WORLD_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv(port_var, str(port))

    module.main()

    assert calls == [(("streamable-http",), {"host": "0.0.0.0", "port": port})]
