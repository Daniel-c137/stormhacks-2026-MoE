"""GitHub and Jira over MCP: the world mocks or the real servers, with the same tool names."""

from typing import Literal

from pydantic import BaseModel

McpServerName = Literal["github", "jira"]


class McpServer(BaseModel):
    name: McpServerName
    url: str


# The orchestrator may call these on its own.
READ_TOOLS: dict[McpServerName, frozenset[str]] = {
    "github": frozenset(
        {
            "list_issues",
            "search_issues",
            "issue_read",
            "list_pull_requests",
            "search_pull_requests",
            "pull_request_read",
            "list_commits",
            "list_releases",
            "get_latest_release",
        }
    ),
    "jira": frozenset(
        {
            "searchJiraIssuesUsingJql",
            "getJiraIssue",
            "getTransitionsForJiraIssue",
            "lookupJiraAccountId",
        }
    ),
}

# Only the approved task-push flow may call these.
WRITE_TOOLS: dict[McpServerName, frozenset[str]] = {
    "github": frozenset({"add_issue_comment"}),
    "jira": frozenset(
        {
            "createJiraIssue",
            "editJiraIssue",
            "transitionJiraIssue",
            "addCommentToJiraIssue",
        }
    ),
}
