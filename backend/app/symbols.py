from __future__ import annotations

import re


CODE_PATTERN = re.compile(r"(?<!\d)(\d{6})(?!\d)")
MODEL_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def normalize_stock_code(value: object) -> str:
    text = str(value or "").strip().upper()
    match = CODE_PATTERN.search(text)
    if not match:
        raise ValueError(f"Invalid stock code: {value!r}")
    return match.group(1)


def normalize_model_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if len(text) == 6 and text.isdigit():
        return text
    if MODEL_SYMBOL_PATTERN.fullmatch(text):
        return text
    raise ValueError(f"Invalid model code: {value!r}")


def exchange_for_code(code: str) -> str:
    normalized = normalize_stock_code(code)
    if normalized.startswith(("5", "6", "9")):
        return "SH"
    if normalized.startswith(("0", "1", "2", "3")):
        return "SZ"
    if normalized.startswith(("4", "8")):
        return "BJ"
    return "UNKNOWN"


def is_fund_code(code: str) -> bool:
    normalized = normalize_stock_code(code)
    return normalized.startswith(("15", "16", "50", "51", "52", "56", "58"))


def baostock_symbol(code: str) -> str:
    normalized = normalize_stock_code(code)
    return f"{exchange_for_code(normalized).lower()}.{normalized}"


def tushare_symbol(code: str) -> str:
    normalized = normalize_stock_code(code)
    return f"{normalized}.{exchange_for_code(normalized)}"

