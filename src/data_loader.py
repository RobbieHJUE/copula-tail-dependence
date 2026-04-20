"""
Data loading utilities for the copula tail-dependence project.

This module provides pure functions for downloading and loading financial
return data. It has no side effects beyond what's explicitly requested
(e.g., saving to disk only when `save_to` is passed).

Typical usage from a notebook or script:

    >>> from src.data_loader import load_returns
    >>> logret = load_returns()

Or to re-download from scratch:

    >>> from src.data_loader import download_prices, compute_log_returns
    >>> prices = download_prices(["SPY", "EFA"], "2005-01-01", "2024-12-31")
    >>> returns = compute_log_returns(prices)
"""

from __future__ import annotations  # forward references in type hints

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 把路径集中管理,避免散落在各处的魔法字符串
# Path(__file__).resolve().parents[1] 是 src/ 的上一级,即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default asset universe; can be overridden by caller
# 7-asset universe covering US/Intl/EM equity, govt/credit bonds, gold, REITs.
# Start date is HYG inception (2007-04-11); earlier data requires a shorter
# universe.
DEFAULT_TICKERS = ("SPY", "EFA", "EEM", "TLT", "HYG", "VNQ", "GLD")
DEFAULT_START = "2007-04-11"
DEFAULT_END = "2024-12-31"


# 用logger而不是print,让调用方决定要不要看输出
# (比如notebook里可以静音,CLI里可以verbose)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to fetch (e.g., ``["SPY", "EFA"]``).
    start, end : str
        Date range in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index, tickers as columns, prices as values.
        Rows with any missing ticker are dropped (we need aligned panels for
        copula estimation).

    Raises
    ------
    RuntimeError
        If the download returns an empty DataFrame (network issue, delisted
        ticker, or wrong date range).
    """
    tickers = list(tickers)  # freeze iterable so we can reuse it
    logger.info(f"Downloading {tickers} from {start} to {end}")

    # auto_adjust=True applies dividend + split adjustments.
    # progress=False because we use our own logger.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex column df for multiple tickers.
    # We only want the "Close" level.
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw

    # Enforce column order (yfinance doesn't guarantee it)
    prices = prices[tickers]

    # Drop rows where any ticker is missing (e.g., holiday mismatches,
    # ETF inception dates). This is a design choice: for copula work we
    # need a fully-aligned panel.
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError(
            f"Empty dataframe returned for {tickers} [{start} .. {end}]. "
            "Check network connectivity, ticker validity, and date range."
        )

    logger.info(
        f"Downloaded {prices.shape[0]} rows "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )
    return prices


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    Log returns are preferred over simple returns for time-series modeling:

    1. They are approximately normally distributed for small returns.
    2. They are time-additive: r_{1:T} = sum(r_t).
    3. GARCH models are conventionally specified in log returns.

    The first row is dropped (NaN from the shift). Any zero or negative
    price (shouldn't happen for ETFs but we guard anyway) would produce
    -inf or NaN and be dropped.
    """
    # Guard against non-positive prices (data errors). Log of 0 or negative
    # gives -inf / NaN which would silently contaminate downstream results.
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found in {bad}")

    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info(f"Saved {df.shape} to {path}")


def load_returns(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the processed log-return panel.

    This is the main entry point for downstream code (notebooks, model
    scripts). By centralizing the load logic here, we guarantee that
    every piece of analysis sees the same data contract:

    - Index: `pd.DatetimeIndex` named "Date"
    - Columns: tickers (str)
    - Values: daily log returns (float)

    Parameters
    ----------
    path : Path, optional
        CSV path. Defaults to ``data/processed/log_returns.csv``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. Run ``scripts/download_data.py`` first.
    """
    if path is None:
        path = PROCESSED_DIR / "log_returns.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def summarize_returns(logret: pd.DataFrame) -> pd.DataFrame:
    """Return a summary stats table for a log-return panel.

    We care especially about:

    - Excess kurtosis >> 0: evidence of fat tails (Gaussian inadequate).
    - Skewness: captures asymmetry; typically negative for equities.
    """
    return pd.DataFrame({
        "mean(%)":     logret.mean() * 100,
        "std(%)":      logret.std() * 100,
        "skew":        logret.skew(),
        "ex_kurt":     logret.kurtosis(),   # pandas returns excess kurtosis
        "min(%)":      logret.min() * 100,
        "max(%)":      logret.max() * 100,
        "n_obs":       logret.count(),
    }).round(3)


def check_tickers_availability(
    tickers: Iterable[str] = DEFAULT_TICKERS,
) -> pd.DataFrame:
    """Query first available date for each ticker.

    Useful when extending the asset universe: prevents silently truncating
    the sample because some ETF has a later inception.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, first_date, last_date.
    """
    rows = []
    for t in tickers:
        # period="max" fetches full history; we only need first/last dates
        hist = yf.Ticker(t).history(period="max")
        if hist.empty:
            rows.append({"ticker": t, "first_date": None, "last_date": None})
        else:
            rows.append({
                "ticker": t,
                "first_date": hist.index.min().date(),
                "last_date": hist.index.max().date(),
            })
    return pd.DataFrame(rows)