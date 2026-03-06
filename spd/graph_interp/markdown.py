"""Minimal Markdown document builder for prompt construction."""


class Md:
    """Accumulates Markdown lines with a fluent API.

    Each method appends content and returns self for chaining.
    Call .build() to get the final string.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def h2(self, text: str) -> "Md":
        self._parts.append(f"## {text}")
        return self

    def h3(self, text: str) -> "Md":
        self._parts.append(f"### {text}")
        return self

    def p(self, text: str) -> "Md":
        self._parts.append(text)
        return self

    def bullet(self, text: str) -> "Md":
        self._parts.append(f"- {text}")
        return self

    def numbered(self, items: list[str]) -> "Md":
        for i, item in enumerate(items, 1):
            self._parts.append(f"{i}. {item}")
        return self

    def blank(self) -> "Md":
        self._parts.append("")
        return self

    def text(self, raw: str) -> "Md":
        self._parts.append(raw)
        return self

    def build(self) -> str:
        return "\n".join(self._parts)
