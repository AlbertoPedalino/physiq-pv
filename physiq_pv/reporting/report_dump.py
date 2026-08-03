"""Print notebook results as one plain-text block that can be copied out.

Notebook rich output does not survive a copy-paste into a chat or an email:
tables lose their alignment and long frames are silently truncated. This
renders a sequence of results as delimited CSV instead, which stays compact,
keeps full precision and can be pasted back into pandas.
"""
from __future__ import annotations

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
    return frame.to_csv(index=False, float_format=float_format).rstrip() + note


def dump_sections(
    sections: Mapping[str, Any],
    *,
    max_rows: Optional[int] = 40,
    float_format: str = "%.4f",
    width: int = 72,
) -> None:
    """Print each named result as a delimited CSV block."""
    for title, value in sections.items():
        print("=" * width)
        print(f"## {title}")
        print("=" * width)
        if value is None:
            print("(non disponibile)")
        else:
            print(_render(value, max_rows=max_rows, float_format=float_format))
        print()
