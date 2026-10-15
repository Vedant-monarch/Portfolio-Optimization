# cluster_qpo/data/yfinance_loader.py

import os
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def download_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    cache: bool = True
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    Caches locally to avoid repeated API calls.

    Parameters
    ----------
    tickers : list[str]
    start : str   YYYY-MM-DD
    end   : str   YYYY-MM-DD
    cache : bool  Load from parquet if available

    Returns
    -------
    pd.DataFrame
        Rows = dates, Columns = tickers
    """
    cache_file = CACHE_DIR / f"prices_{start}_{end}_{len(tickers)}t.parquet"

    if cache and cache_file.exists():
        print(f"[loader] Loading prices from cache: {cache_file.name}")
        return pd.read_parquet(cache_file)

    print(f"[loader] Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with too many NaNs (>10%)
    threshold = int(0.9 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill().dropna()

    if cache:
        prices.to_parquet(cache_file)
        print(f"[loader] Cached to: {cache_file.name}")

    return prices


def compute_returns(
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute daily percentage returns from prices.

    Parameters
    ----------
    prices : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Daily returns, first row dropped
    """
    return prices.pct_change().dropna()


def load_returns(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-01-01",
    cache: bool = True
) -> pd.DataFrame:
    """
    One-call convenience function.
    Downloads prices and computes returns.

    Parameters
    ----------
    tickers : list[str]
    start   : str
    end     : str
    cache   : bool

    Returns
    -------
    pd.DataFrame
        Daily returns
    """
    prices = download_prices(
        tickers=tickers,
        start=start,
        end=end,
        cache=cache
    )
    returns = compute_returns(prices)
    print(f"[loader] Returns shape: {returns.shape}")
    return returns
