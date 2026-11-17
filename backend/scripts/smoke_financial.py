from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.baostock import BaostockProvider
from app.symbols import baostock_symbol


QUERIES = {
    "profit": "query_profit_data",
    "operation": "query_operation_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "cash_flow": "query_cash_flow_data",
    "dupont": "query_dupont_data",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BaoStock financial fields")
    parser.add_argument("--code", default="000001")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--quarter", type=int, default=4)
    args = parser.parse_args()

    provider = BaostockProvider()
    bs = provider._login()
    try:
        for label, method_name in QUERIES.items():
            method = getattr(bs, method_name)
            result = method(
                code=baostock_symbol(args.code),
                year=args.year,
                quarter=args.quarter,
            )
            rows: list[list[str]] = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            print(f"\n[{label}] error={result.error_code} {result.error_msg}")
            print(f"fields={result.fields}")
            print(f"rows={rows}")
    finally:
        bs.logout()


if __name__ == "__main__":
    main()

