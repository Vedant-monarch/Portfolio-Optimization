# qpo/clustering/backtest.py

import numpy as np
import pandas as pd
from .reduction import build_reduced_portfolio_inputs


def equal_weight_portfolio(n_assets: int) -> pd.Series:
    """
    Return equal-weight portfolio as a pd.Series.

    Parameters
    ----------
    n_assets : int

    Returns
    -------
    pd.Series
        Weights summing to 1.0
    """
    return pd.Series(
        np.ones(n_assets) / n_assets,
        name="weight"
    )


def portfolio_metrics(
    returns: pd.DataFrame,
    weights: pd.Series = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Compute portfolio performance metrics
    for a given universe and weights.

    If no weights provided, uses equal weighting.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows=dates, cols=assets)
    weights : pd.Series, optional
        Asset weights. Defaults to equal weight.
    risk_free_rate : float
        Annualised risk-free rate

    Returns
    -------
    dict with keys:
        - "expected_return"      : float (annualised)
        - "volatility"           : float (annualised)
        - "sharpe_ratio"         : float
        - "max_drawdown"         : float
        - "diversification_ratio": float
    """
    if weights is None:
        n = len(returns.columns)
        weights = pd.Series(
            np.ones(n) / n,
            index=returns.columns
        )

    # Align weights to returns columns
    weights = weights.reindex(returns.columns).fillna(0)
    weights = weights / weights.sum()

    # Portfolio daily returns
    port_returns = returns.dot(weights)

    # Annualised expected return
    expected_return = float(port_returns.mean() * 252)

    # Annualised volatility
    volatility = float(port_returns.std() * np.sqrt(252))

    # Sharpe ratio
    sharpe = (
        (expected_return - risk_free_rate) / volatility
        if volatility > 0 else 0.0
    )

    # Maximum drawdown
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Diversification ratio
    # DR = (w' * sigma_i) / sqrt(w' * Sigma * w)
    asset_vols = returns.std() * np.sqrt(252)
    weighted_avg_vol = float(weights.dot(asset_vols))
    diversification_ratio = (
        weighted_avg_vol / volatility
        if volatility > 0 else 1.0
    )

    return {
        "expected_return": round(expected_return, 6),
        "volatility": round(volatility, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "diversification_ratio": round(diversification_ratio, 4)
    }


def backtest_representative_universe(
    returns: pd.DataFrame,
    n_clusters: int = 20,
    representative_method: str = "centroid",
    risk_free_rate: float = 0.0
) -> dict:
    """
    Run portfolio metrics on the
    cluster-reduced representative universe.

    Parameters
    ----------
    returns : pd.DataFrame
    n_clusters : int
    representative_method : str
    risk_free_rate : float

    Returns
    -------
    dict with keys:
        - "metrics"         : dict
        - "reduced_returns" : pd.DataFrame
        - "representatives" : pd.DataFrame
        - "clusters"        : dict
    """
    result = build_reduced_portfolio_inputs(
        returns=returns,
        n_clusters=n_clusters,
        representative_method=representative_method
    )

    reduced_returns = result["returns"]
    metrics = portfolio_metrics(
        reduced_returns,
        risk_free_rate=risk_free_rate
    )

    return {
        "metrics": metrics,
        "reduced_returns": reduced_returns,
        "representatives": result["representatives"],
        "clusters": result["clusters"]
    }


def compare_reduced_vs_full(
    returns: pd.DataFrame,
    n_clusters: int = 20,
    representative_method: str = "centroid",
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Compare portfolio metrics between:
        - Full universe (equal weight)
        - Cluster-reduced universe (equal weight)

    Parameters
    ----------
    returns : pd.DataFrame
    n_clusters : int
    representative_method : str
    risk_free_rate : float

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison table
    """
    # Full universe metrics
    full_metrics = portfolio_metrics(
        returns,
        risk_free_rate=risk_free_rate
    )

    # Reduced universe metrics
    reduced_result = backtest_representative_universe(
        returns=returns,
        n_clusters=n_clusters,
        representative_method=representative_method,
        risk_free_rate=risk_free_rate
    )
    reduced_metrics = reduced_result["metrics"]

    # Build comparison table
    comparison = pd.DataFrame({
        "Full Universe": full_metrics,
        "Reduced Universe": reduced_metrics
    })

    comparison["Difference"] = (
        comparison["Reduced Universe"]
        - comparison["Full Universe"]
    ).round(6)

    comparison["% Change"] = (
        (comparison["Difference"]
         / comparison["Full Universe"].abs())
        * 100
    ).round(2)

    return comparison
