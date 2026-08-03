"""Print notebook results as one plain-text block that can be copied out.

Notebook rich output does not survive a copy-paste into a chat or an email:
tables lose their alignment and long frames are silently truncated. This
renders a sequence of results as delimited CSV instead, which stays compact,
keeps full precision and can be pasted back into pandas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


def _render(value: Any, *, max_rows: Optional[int], float_format: str) -> str:
    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name or "value")
    if isinstance(value, Mapping):
        value = pd.DataFrame(
            {"key": list(value), "value": list(value.values())}
        )
    if not isinstance(value, pd.DataFrame):
        return str(value)
    frame = value
    note = ""
    if max_rows is not None and len(frame) > max_rows:
        note = f"\n# ...{len(frame) - max_rows} righe omesse su {len(frame)}"
        frame = frame.head(max_rows)
    if frame.index.name is not None or not isinstance(
        frame.index, pd.RangeIndex
    ):
        frame = frame.reset_index()
    # to_csv defaults to os.linesep when rendering to a string, which would put
    # CRLF inside a file meant to be read on another platform.
    rendered = frame.to_csv(
        index=False, float_format=float_format, lineterminator="\n"
    )
    return rendered.rstrip() + note


def dump_sections(
    sections: Mapping[str, Any],
    *,
    path: Optional[str | Path] = None,
    max_rows: Optional[int] = 40,
    float_format: str = "%.4f",
    width: int = 72,
) -> Optional[Path]:
    """Print each named result as a delimited CSV block.

    With ``path`` the same text is written to a file, so the results can be
    copied off the machine in one transfer instead of being selected out of the
    notebook by hand.
    """
    blocks = []
    for title, value in sections.items():
        rendered = (
            "(non disponibile)"
            if value is None
            else _render(value, max_rows=max_rows, float_format=float_format)
        )
        blocks.append(f"{'=' * width}\n## {title}\n{'=' * width}\n{rendered}\n")
    text = "\n".join(blocks)
    print(text)
    if path is None:
        return None
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Explicit newline: the file is copied between machines, and platform
    # translation would double the line breaks on the way back.
    resolved.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dump] scritto: {resolved}")
    return resolved
