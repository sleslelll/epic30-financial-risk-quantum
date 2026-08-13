"""
EPIC30 - Financial Risk & Quantum Optimization

Market data loader for the financial network sandbox.

This module:
1. Downloads market data
2. Aligns observations by date
3. Computes log returns
4. Produces a clean return matrix for downstream
   covariance, graph, and spectral analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Market universe
# ---------------------------------------------------------

TICKERS = {
    "ETH": "ETH-USD",
    "BTC": "BTC-USD",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "GOLD": "GC=F",
    "USD": "DX-Y.NYB",
    "VIX": "^VIX",
}


# ---------------------------------------------------------
# Download market data
# ---------------------------------------------------------

def download_prices(
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted market prices from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Columns represent financial assets/factors.
        Rows represent trading dates.
    """

    raw = yf.download(
        list(TICKERS.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"].copy()

    # Convert Yahoo ticker names into readable names
    reverse_tickers = {v: k for k, v in TICKERS.items()}
    prices = prices.rename(columns=reverse_tickers)

    # Keep consistent column order
    prices = prices[list(TICKERS.keys())]

    # Crypto trades 7 days/week while traditional markets do not.
    # Keep only dates for which all selected markets have observations.
    prices = prices.dropna()

    return prices


# ---------------------------------------------------------
# Return transformation
# ---------------------------------------------------------

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices into daily log returns.

    r_t = log(P_t / P_{t-1})
    """

    returns = np.log(prices / prices.shift(1))

    return returns.dropna()


# ---------------------------------------------------------
# Complete dataset
# ---------------------------------------------------------

def build_return_matrix(
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Build the aligned return matrix used by later EPIC30 modules.
    """

    prices = download_prices(start=start, end=end)
    returns = calculate_log_returns(prices)

    return returns


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

def save_dataset(
    returns: pd.DataFrame,
    filename: str = "market_returns.csv",
) -> Path:
    """
    Save processed return data to the project's data directory.
    """

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / filename
    returns.to_csv(output_path)

    return output_path


# ---------------------------------------------------------
# Run 
# ---------------------------------------------------------

if __name__ == "__main__":

    returns = build_return_matrix()

    print("\nEPIC30 Market Return Matrix")
    print("---------------------------")
    print(returns.head())

    print("\nDataset shape:")
    print(returns.shape)

    print("\nCorrelation with ETH:")
    print(
        returns.corr()["ETH"]
        .sort_values(ascending=False)
    )

    path = save_dataset(returns)

    print(f"\nDataset saved to: {path}")