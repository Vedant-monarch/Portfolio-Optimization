# qpo/clustering/reduction.py

import pandas as pd
from .pipeline import run_clustering_reduction
from .correlation import (
    covariance_matrix,
    expected_returns
)


def build_reduced_portfolio_inputs(
    returns: pd.DataFrame,
    n_clusters: int = 20,
    representative_method: str = "centroid"
) -> dict:
    """
    Reduce a large universe into a smaller
    representative universe suitable for QUBO.

    Parameters
    ----------
    returns : pd.DataFrame
        Full universe returns (rows=dates, cols=assets)
    n_clusters : int
        Number of clusters to form
    representative_method : str
        "centroid" or "sharpe"

    Returns
    -------
    dict with keys:
        - "mu"              : pd.Series  (expected returns)
        - "sigma"           : pd.DataFrame (covariance matrix)
        - "returns"         : pd.DataFrame (reduced returns)
        - "representatives" : pd.DataFrame
        - "clusters"        : dict[int, list[str]]
    """
    clustering = run_clustering_reduction(
        returns=returns,
        n_clusters=n_clusters,
        method=representative_method
    )

    reduced_returns = clustering["reduced_returns"]

    mu = expected_returns(reduced_returns)
    sigma = covariance_matrix(reduced_returns)

    return {
        "mu": mu,
        "sigma": sigma,
        "returns": reduced_returns,
        "representatives": clustering["representatives"],
        "clusters": clustering["clusters"]
    }
