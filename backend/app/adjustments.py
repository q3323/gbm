from __future__ import annotations

from typing import Final


ADJUST_TYPES: Final = ("raw", "qfq", "hfq")
DEFAULT_ADJUST_TYPE: Final = "qfq"

_ALIASES = {
    "": "raw",
    "none": "raw",
    "raw": "raw",
    "qfq": "qfq",
    "hfq": "hfq",
}


def normalize_adjust_type(
    value: str | None,
    *,
    default: str = DEFAULT_ADJUST_TYPE,
) -> str:
    text = (value or "").strip().lower()
    if not text:
        text = default
    normalized = _ALIASES.get(text)
    if normalized is None:
        raise ValueError(f"Unsupported adjustment type: {value!r}")
    return normalized
