# qpo/clustering/clustered_portfolio.py

import pandas as pd
from .reduction import build_reduced_portfolio_inputs


class ClusteredPortfolioProblem:
    """
    Portfolio optimization problem built from
    correlation-clustered representative assets.

    Pipeline
    --------
    Full Universe
        ↓
    Clustering
        ↓
    Representatives
        ↓
    μ, Σ
        ↓
    QUBO
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        n_clusters: int = 20,
        representative_method: str = "centroid"
    ):
        self.returns = returns
        self.n_clusters = n_clusters
        self.representative_method = (
            representative_method
        )
        self._build()

    def _build(self):
        result = build_reduced_portfolio_inputs(
            returns=self.returns,
            n_clusters=self.n_clusters,
            representative_method=self.representative_method
        )

        self.mu = result["mu"]
        self.sigma = result["sigma"]
        self.reduced_returns = result["returns"]
        self.representatives = result["representatives"]
        self.clusters = result["clusters"]

    def summary(self) -> dict:
        """
        Return high-level summary of the
        clustering reduction.

        Returns
        -------
        dict with keys:
            - original_assets
            - reduced_assets
            - n_clusters
            - representative_method
            - compression_ratio
        """
        original = self.returns.shape[1]
        reduced = self.reduced_returns.shape[1]

        return {
            "original_assets": original,
            "reduced_assets": reduced,
            "n_clusters": self.n_clusters,
            "representative_method": self.representative_method,
            "compression_ratio": round(reduced / original, 4)
        }

    def get_mu(self) -> pd.Series:
        """Return expected returns vector (annualised)."""
        return self.mu

    def get_sigma(self) -> pd.DataFrame:
        """Return covariance matrix of reduced universe."""
        return self.sigma

    def get_asset_names(self) -> list[str]:
        """Return list of representative asset tickers."""
        return list(self.reduced_returns.columns)

    def get_representatives(self) -> pd.DataFrame:
        """Return representative asset per cluster."""
        return self.representatives

    def get_cluster_mapping(self) -> dict[int, list[str]]:
        """Return full cluster → assets mapping."""
        return self.clusters

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ClusteredPortfolioProblem("
            f"original={s['original_assets']} assets, "
            f"reduced={s['reduced_assets']} assets, "
            f"n_clusters={s['n_clusters']}, "
            f"method='{s['representative_method']}', "
            f"compression={s['compression_ratio']})"
        )
