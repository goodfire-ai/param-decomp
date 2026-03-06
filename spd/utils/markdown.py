"""Minimal Markdown document builder for prompt construction.

Atomic unit is a block (paragraph, heading, list, etc.).
build() joins blocks with double newlines.
"""


class Md:
    """Accumulates Markdown blocks with a fluent API.

    Each method appends a block and returns self for chaining.
    Call .build() to get the final string (blocks joined by blank lines).
    """

    def __init__(self) -> None:
        self._blocks: list[str] = []

    def h(self, level: int, text: str) -> "Md":
        self._blocks.append(f"{'#' * level} {text}")
        return self

    def p(self, text: str) -> "Md":
        self._blocks.append(text)
        return self

    def bullets(self, items: list[str]) -> "Md":
        self._blocks.append("\n".join(f"- {item}" for item in items))
        return self

    def labeled_list(self, label: str, items: list[str]) -> "Md":
        lines = [label] + [f"- {item}" for item in items]
        self._blocks.append("\n".join(lines))
        return self

    def numbered(self, items: list[str]) -> "Md":
        self._blocks.append("\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)))
        return self

    def extend(self, other: "Md") -> "Md":
        self._blocks.extend(other._blocks)
        return self

    def build(self) -> str:
        return "\n\n".join(self._blocks)
