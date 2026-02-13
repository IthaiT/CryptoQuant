"""Convert a .km (JSON mindmap) file into a Markdown outline.

- Traverses all nodes in order (no omissions by design).
- Uses headings for top levels and nested bullet lists for deeper levels.

Usage:
  python script/km_to_md.py --input docs/notes/2.km --output docs/notes/2.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_children(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    children = node.get("children", [])
    if not isinstance(children, list):
        return []
    for child in children:
        if isinstance(child, dict):
            yield child


def _node_text(node: dict[str, Any]) -> str:
    data = node.get("data", {})
    if not isinstance(data, dict):
        return ""
    text = data.get("text", "")
    return str(text).strip()


def _escape_md(text: str) -> str:
    # Minimal escaping to prevent accidental markdown structure breaks.
    # Keep it conservative; we want content unchanged as much as possible.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def render_node(node: dict[str, Any], depth: int) -> list[str]:
    """Render node and all descendants.

    depth:
      0 = root
      1 = part level
      2 = subpart level
      >=3 = bullets
    """

    text = _escape_md(_node_text(node))
    lines: list[str] = []

    # Decide formatting by depth.
    if depth == 0:
        if text:
            lines.append(f"# {text}")
            lines.append("")
    elif depth == 1:
        if text:
            lines.append(f"## {text}")
            lines.append("")
    elif depth == 2:
        if text:
            lines.append(f"### {text}")
            lines.append("")
    else:
        # Bullet list for deeper levels.
        if text:
            indent = "  " * (depth - 3)
            lines.append(f"{indent}- {text}")

    # Render children.
    child_blocks: list[str] = []
    for child in _iter_children(node):
        child_lines = render_node(child, depth + 1)
        child_blocks.extend(child_lines)

    # For bullet sections, keep children directly below; for headings, keep spacing.
    if depth >= 3:
        lines.extend(child_blocks)
    else:
        if child_blocks:
            lines.extend(child_blocks)
            # Ensure a blank line after each section for readability.
            if lines and lines[-1] != "":
                lines.append("")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .km JSON mindmap to Markdown")
    parser.add_argument("--input", required=True, help="Input .km file path")
    parser.add_argument("--output", required=True, help="Output .md file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "root" not in data:
        raise SystemExit("Invalid .km JSON: expected top-level object with 'root'")

    root = data["root"]
    if not isinstance(root, dict):
        raise SystemExit("Invalid .km JSON: 'root' must be an object")

    md_lines = render_node(root, 0)

    # Normalize: collapse excessive blank lines (max 2).
    normalized: list[str] = []
    blank_run = 0
    for line in md_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                normalized.append("")
        else:
            blank_run = 0
            normalized.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(normalized).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
