import pytest

from app.symbols import (
    baostock_symbol,
    exchange_for_code,
    normalize_model_code,
    normalize_stock_code,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("000001", "000001"),
        ("sz.000001", "000001"),
        ("600000.SH", "600000"),
        (430047, "430047"),
    ],
)
def test_normalize_stock_code(raw, expected) -> None:
    assert normalize_stock_code(raw) == expected


def test_exchange_and_baostock_symbol() -> None:
    assert exchange_for_code("600000") == "SH"
    assert baostock_symbol("000001") == "sz.000001"


def test_rejects_invalid_code() -> None:
    with pytest.raises(ValueError):
        normalize_stock_code("not-a-code")


def test_model_code_accepts_us_tickers() -> None:
    assert normalize_model_code("qqq") == "QQQ"
    assert normalize_model_code("QQQM") == "QQQM"
    assert normalize_model_code("000001") == "000001"

