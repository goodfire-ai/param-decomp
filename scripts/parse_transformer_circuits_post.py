#!/usr/bin/env python3
"""Export a Transformer Circuits post to local Markdown + downloaded image assets.

Example:
  .venv/bin/python scripts/parse_transformer_circuits_post.py \
    --url https://transformer-circuits.pub/2025/attribution-graphs/biology.html \
    --output-md papers/biology_source/biology.md \
    --assets-dir papers/biology_source/assets
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

DEFAULT_URL = "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"
USER_AGENT = "spd-biology-exporter/1.0"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def should_insert_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left = left.rstrip()
    right = right.lstrip()
    if not left or not right:
        return False
    if left.endswith((" ", "\n", "\t", "/", "(", "[", "{", "-", "“", '"', "'")):
        return False
    if right.startswith(
        (" ", "\n", "\t", ".", ",", ":", ";", "!", "?", ")", "]", "}", "-", "”", '"', "'")
    ):
        return False
    if left.endswith("  "):
        return False
    left_char = left[-1]
    right_char = right[0]
    return bool(
        re.match(r"[A-Za-z0-9\]\)]", left_char) and re.match(r"[A-Za-z0-9\[\(]", right_char)
    )


class Exporter:
    def __init__(
        self,
        *,
        base_url: str,
        output_md: Path,
        assets_dir: Path,
        download_assets: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.output_md = output_md
        self.assets_dir = assets_dir
        self.download_assets = download_assets
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.asset_map: dict[str, str] = {}
        self.asset_counter = 0

    def fetch_html(self) -> str:
        response = self.session.get(self.base_url, timeout=self.timeout)
        response.raise_for_status()
        # Distill pages sometimes produce mojibake via default charset guessing.
        return response.content.decode("utf-8", errors="replace")

    def _relative_asset_path(self, local_path: Path) -> str:
        return local_path.relative_to(self.output_md.parent).as_posix()

    def _unique_asset_filename(self, remote_url: str) -> str:
        parsed = urlparse(remote_url)
        basename = Path(parsed.path).name or f"asset_{self.asset_counter}"
        if "." not in basename:
            basename = f"{basename}.bin"
        candidate = basename
        while (self.assets_dir / candidate).exists():
            self.asset_counter += 1
            stem = Path(basename).stem
            suffix = Path(basename).suffix
            candidate = f"{stem}_{self.asset_counter}{suffix}"
        return candidate

    def download_asset(self, src: str) -> str:
        if not src:
            return ""
        remote_url = urljoin(self.base_url, src)
        if remote_url in self.asset_map:
            return self.asset_map[remote_url]
        if remote_url.startswith("data:"):
            return remote_url
        local_rel = remote_url
        if self.download_assets:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            preferred = Path(urlparse(remote_url).path).name
            if preferred and "." in preferred and (self.assets_dir / preferred).exists():
                local_path = self.assets_dir / preferred
            else:
                filename = self._unique_asset_filename(remote_url)
                local_path = self.assets_dir / filename
                response = self.session.get(remote_url, timeout=self.timeout)
                response.raise_for_status()
                local_path.write_bytes(response.content)
            local_rel = self._relative_asset_path(local_path)
        self.asset_map[remote_url] = local_rel
        return local_rel

    def render_inline(self, node: Tag | NavigableString) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        name = node.name.lower()
        if name == "br":
            return "  \n"
        if name in {"b", "strong"}:
            return f"**{self.render_children_inline(node.children)}**"
        if name in {"i", "em"}:
            return f"*{self.render_children_inline(node.children)}*"
        if name == "code":
            text = normalize_whitespace(self.render_children_inline(node.children))
            return f"`{text}`" if text else ""
        if name == "a":
            href = (node.get("href") or "").strip()
            text = normalize_whitespace(self.render_children_inline(node.children))
            if not text:
                text = href
            if not href:
                return text
            full_href = urljoin(self.base_url, href)
            return f"[{text}]({full_href})"
        if name == "d-cite":
            cite = normalize_whitespace(self.render_children_inline(node.children))
            return f"[{cite}]" if cite else "[citation]"
        if name == "d-footnote":
            note = normalize_whitespace(self.render_children_inline(node.children))
            return f" (Footnote: {note}) " if note else ""
        return self.render_children_inline(node.children)

    def render_children_inline(self, nodes: Iterable[Tag | NavigableString]) -> str:
        pieces = [self.render_inline(child) for child in nodes]
        raw = ""
        for piece in pieces:
            if not piece:
                continue
            if raw and should_insert_space(raw, piece):
                raw += " "
            raw += piece
        # Keep explicit markdown line breaks but normalize other whitespace.
        parts = raw.split("  \n")
        return "  \n".join(normalize_whitespace(part) for part in parts)

    def render_list(self, list_tag: Tag, level: int = 0) -> list[str]:
        lines: list[str] = []
        ordered = list_tag.name.lower() == "ol"
        counter = 1
        indent = "  " * level
        for li in list_tag.find_all("li", recursive=False):
            inline_nodes: list[Tag | NavigableString] = []
            nested_lists: list[Tag] = []
            for child in li.children:
                if isinstance(child, Tag) and child.name and child.name.lower() in {"ul", "ol"}:
                    nested_lists.append(child)
                else:
                    inline_nodes.append(child)
            text = normalize_whitespace(self.render_children_inline(inline_nodes))
            marker = f"{counter}." if ordered else "-"
            if text:
                lines.append(f"{indent}{marker} {text}")
            else:
                lines.append(f"{indent}{marker}")
            for nested in nested_lists:
                lines.extend(self.render_list(nested, level + 1))
            counter += 1
        lines.append("")
        return lines

    def render_table(self, table: Tag) -> list[str]:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            row: list[str] = []
            cells = tr.find_all(["th", "td"])
            for cell in cells:
                row.append(normalize_whitespace(self.render_children_inline(cell.children)))
            if row:
                rows.append(row)
        if not rows:
            return []
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        header = padded[0]
        sep = ["---"] * width
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in padded[1:]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return lines

    def render_figure(self, figure: Tag) -> list[str]:
        lines: list[str] = []
        imgs = figure.find_all("img")
        for img in imgs:
            src = (img.get("src") or "").strip()
            alt = normalize_whitespace(img.get("alt") or "")
            local_src = self.download_asset(src)
            if local_src:
                lines.append(f"![{alt}]({local_src})")
        caption = figure.find("figcaption")
        if caption:
            caption_text = normalize_whitespace(self.render_children_inline(caption.children))
            if caption_text:
                lines.append(f"_Figure: {caption_text}_")
        if lines:
            lines.append("")
        return lines

    def render_block(self, node: Tag | NavigableString) -> list[str]:
        if isinstance(node, NavigableString):
            text = normalize_whitespace(str(node))
            return [text, ""] if text else []
        if not isinstance(node, Tag):
            return []
        name = node.name.lower()
        if name in {"style", "script"}:
            return []
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = normalize_whitespace(self.render_children_inline(node.children))
            if not text:
                return []
            anchor = node.get("id")
            anchor_suffix = f" <!-- id: {anchor} -->" if anchor else ""
            return [f"{'#' * level} {text}{anchor_suffix}", ""]
        if name == "p":
            text = normalize_whitespace(self.render_children_inline(node.children))
            return [text, ""] if text else []
        if name in {"ul", "ol"}:
            return self.render_list(node)
        if name == "figure":
            return self.render_figure(node)
        if name == "table":
            return self.render_table(node)
        if name == "hr":
            return ["---", ""]
        if name == "br":
            return [""]
        if name == "d-contents":
            return []
        if name in {"div", "section", "nav", "d-appendix", "d-article"}:
            lines: list[str] = []
            for child in node.children:
                lines.extend(self.render_block(child))
            return lines
        text = normalize_whitespace(self.render_children_inline(node.children))
        return [text, ""] if text else []

    def export(self, include_appendix: bool = True) -> tuple[str, int]:
        html = self.fetch_html()
        soup = BeautifulSoup(html, "html.parser")
        title = normalize_whitespace(soup.title.string if soup.title else "") or "Untitled"
        article = soup.find("d-article")
        if article is None:
            raise RuntimeError("Could not find <d-article> in the page")
        lines: list[str] = [
            f"# {title}",
            "",
            f"Source: {self.base_url}",
            "",
            "> Auto-generated by scripts/parse_transformer_circuits_post.py",
            "",
        ]
        for child in article.children:
            lines.extend(self.render_block(child))
        if include_appendix:
            appendix = soup.find("d-appendix")
            if appendix is not None:
                appendix_lines: list[str] = []
                for child in appendix.children:
                    appendix_lines.extend(self.render_block(child))
                appendix_content = [line for line in appendix_lines if line.strip()]
                if appendix_content:
                    lines.extend(["## Appendix", ""])
                    lines.extend(appendix_lines)
        # Strip trailing whitespace and collapse excessive blank lines.
        cleaned: list[str] = []
        blank_run = 0
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                blank_run += 1
                if blank_run <= 1:
                    cleaned.append("")
            else:
                blank_run = 0
                cleaned.append(stripped)
        markdown = "\n".join(cleaned).strip() + "\n"
        self.output_md.parent.mkdir(parents=True, exist_ok=True)
        self.output_md.write_text(markdown, encoding="utf-8")
        return markdown, len(self.asset_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Post URL to parse")
    parser.add_argument(
        "--output-md",
        default="papers/biology_source/biology.md",
        help="Path for generated markdown",
    )
    parser.add_argument(
        "--assets-dir",
        default="papers/biology_source/assets",
        help="Directory for downloaded assets",
    )
    parser.add_argument(
        "--skip-appendix",
        action="store_true",
        help="Do not include d-appendix content",
    )
    parser.add_argument(
        "--skip-download-assets",
        action="store_true",
        help="Keep remote asset links instead of downloading files",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_md = Path(args.output_md).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    exporter = Exporter(
        base_url=args.url,
        output_md=output_md,
        assets_dir=assets_dir,
        download_assets=not args.skip_download_assets,
        timeout=args.timeout,
    )
    _, asset_count = exporter.export(include_appendix=not args.skip_appendix)
    print(f"Wrote markdown: {output_md}")
    if args.skip_download_assets:
        print("Assets were not downloaded (--skip-download-assets set).")
    else:
        print(f"Downloaded/linked assets: {asset_count}")
        print(f"Assets dir: {assets_dir}")


if __name__ == "__main__":
    main()
