"""Small text helpers shared by the agent and the report, with no imports of their own."""


def one_line(text: str) -> str:
    """Whitespace collapsed, wrapping quotes and a trailing period dropped."""
    line = " ".join(text.split()).strip("\"'` \u201c\u201d\u2018\u2019")
    return line[:-1].rstrip() if line.endswith(".") and not line.endswith("..") else line
