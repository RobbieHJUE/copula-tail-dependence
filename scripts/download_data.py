"""
Download price data and produce the processed log-return panel.

Run from the project root:

    python scripts/download_data.py

This script is idempotent: running it multiple times overwrites the output
files with fresh data. It's the *only* place we hit the network, so all
downstream code can assume data is already on disk.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 把项目根加到sys.path,这样脚本能 `from src...` import
# 否则 python scripts/download_data.py 会报 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402  (import after path manipulation)
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    PROCESSED_DIR,
    RAW_DIR,
    compute_log_returns,
    download_prices,
    save_dataframe,
    summarize_returns,
)


def configure_logging() -> None:
    """Set up logging so the user sees progress messages.

    We do this here (in the script) rather than in the library, because
    libraries shouldn't dictate logging config to their callers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    configure_logging()
    log = logging.getLogger("download_data")

    # --- 0. (Optional) Sanity-check ticker inception dates ---
    # Comment this out once you've verified the universe; it's slow-ish
    # (one HTTP call per ticker).
    log.info("Checking ticker availability...")
    from src.data_loader import check_tickers_availability
    avail = check_tickers_availability(DEFAULT_TICKERS)
    print()
    print(avail.to_string(index=False))
    print()

    # --- 1. Download ---
    prices = download_prices(
        tickers=DEFAULT_TICKERS,
        start=DEFAULT_START,
        end=DEFAULT_END,
    )
    save_dataframe(prices, RAW_DIR / "prices.csv")

    # --- 2. Transform ---
    logret = compute_log_returns(prices)
    save_dataframe(logret, PROCESSED_DIR / "log_returns.csv")

    # --- 3. Sanity check ---
    log.info("Summary of log returns:")
    summary = summarize_returns(logret)
    summary.to_csv(PROJECT_ROOT / "results" / "tables" / "table1_summary_stats.csv", index=False)
    print()
    print(summary.to_string())
    print()

    min_kurt = summary["ex_kurt"].min()
    if min_kurt < 1:
        log.warning(
            f"Lowest excess kurtosis = {min_kurt:.2f}. "
            "Expected > 1 for daily equity-like returns; double-check data."
        )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    # `sys.exit` ensures the shell sees our return code.
    # Useful for CI/CD and for `make` targets later.
    sys.exit(main())