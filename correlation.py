# qpo/clustering/correlation.py

import numpy as np
import pandas as pd


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute asset covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows = timestamps
        Columns = assets

    Returns
    -------
    pd.DataFrame
        Covariance matrix
    """
    return returns.cov()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix.

    Parameters
    ----------
    returns : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return returns.corr()


def correlation_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation matrix to HRP distance matrix.

    Formula:
        d(i,j) = sqrt((1 - corr(i,j))/2)

    Parameters
    ----------
    corr : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    distance = np.sqrt((1.0 - corr) / 2.0)

    return pd.DataFrame(
        distance,
        index=corr.index,
        columns=corr.columns
    )


def expected_returns(returns: pd.DataFrame) -> pd.Series:
    """
    Compute mean returns.

    Parameters
    ----------
    returns : pd.DataFrame

    Returns
    -------
    pd.Series
        Mean return per asset
    """
    return returns.mean()


def annualized_volatility(
    returns: pd.DataFrame,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Annualized asset volatility.

    Parameters
    ----------
    returns : pd.DataFrame
    periods_per_year : int

    Returns
    -------
    pd.Series
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_like_scores(
    returns: pd.DataFrame,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Simple Sharpe-like ranking metric.

    Mean annualized return divided by
    annualized volatility.

    Risk-free rate omitted intentionally.

    Parameters
    ----------
    returns : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    mu = returns.mean() * periods_per_year
    sigma = returns.std() * np.sqrt(periods_per_year)

    scores = mu / sigma

    return scores.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)