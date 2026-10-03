from typing import Any, Protocol

from pydantic import BaseModel


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ToolResult(BaseModel):
    name: str
    ok: bool
    content: Any = None
    error: str | None = None


class Toolbox(Protocol):
    """Every tool the orchestrator may call for one team: meeting memory, GitHub and Jira reads."""

    def specs(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
