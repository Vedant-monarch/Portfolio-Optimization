# qpo/cluster_qpo/benchmark.py

import pandas as pd
from .backtest import (
    portfolio_metrics,
    equal_weight_portfolio
)


def benchmark_reduction(
    full_returns: pd.DataFrame,
    reduced_returns: pd.DataFrame
) -> dict:
    """
    Compare full universe against
    reduced universe using equal-weight portfolios.

    Parameters
    ----------
    full_returns : pd.DataFrame
    reduced_returns : pd.DataFrame

    Returns
    -------
    dict with keys:
        - "full_universe"     : dict of metrics
        - "reduced_universe"  : dict of metrics
    """
    full_weights = equal_weight_portfolio(
        full_returns.shape[1]
    )
    full_weights.index = full_returns.columns

    reduced_weights = equal_weight_portfolio(
        reduced_returns.shape[1]
    )
    reduced_weights.index = reduced_returns.columns

    full_metrics = portfolio_metrics(
        full_returns,
        weights=full_weights
    )
    reduced_metrics = portfolio_metrics(
        reduced_returns,
        weights=reduced_weights
    )

    return {
        "full_universe": full_metrics,
        "reduced_universe": reduced_metrics
    }


def universe_statistics(
    full_returns: pd.DataFrame,
    reduced_returns: pd.DataFrame
) -> dict:
    """
    Basic universe comparison statistics.

    Returns
    -------
    dict
    """
    return {
        "full_assets": int(full_returns.shape[1]),
        "reduced_assets": int(reduced_returns.shape[1]),
        "compression_ratio": round(
            float(reduced_returns.shape[1] / full_returns.shape[1]),
            4
        ),
        "assets_removed": int(
            full_returns.shape[1] - reduced_returns.shape[1]
        ),
        "reduction_pct": round(
            (1 - reduced_returns.shape[1] / full_returns.shape[1]) * 100,
            2
        )
    }


def benchmark_clustering_pipeline(
    full_returns: pd.DataFrame,
    reduced_returns: pd.DataFrame,
    clustering_report: dict
) -> dict:
    """
    Complete clustering benchmark report.

    Combines:
        - Universe statistics
        - Clustering quality metrics
        - Portfolio metrics (full vs reduced)

    Parameters
    ----------
    full_returns : pd.DataFrame
    reduced_returns : pd.DataFrame
    clustering_report : dict
        Output of evaluate_clustering()

    Returns
    -------
    dict with keys:
        - "universe_statistics" : dict
        - "clustering_quality"  : dict (summary metrics)
        - "portfolio_metrics"   : dict
    """
    metrics = benchmark_reduction(
        full_returns,
        reduced_returns
    )

    stats = universe_statistics(
        full_returns,
        reduced_returns
    )

    # Extract scalar summary from clustering report
    if "summary" in clustering_report:
        quality = clustering_report["summary"]["value"].to_dict()
    else:
        quality = clustering_report

    return {
        "universe_statistics": stats,
        "clustering_quality": quality,
        "portfolio_metrics": metrics
    }


def print_benchmark_report(report: dict) -> None:
    """
    Human-readable benchmark output.

    Parameters
    ----------
    report : dict
        Output of benchmark_clustering_pipeline()
    """
    print("\n")
    print("=" * 60)
    print("CLUSTERING BENCHMARK REPORT")
    print("=" * 60)

    print("\n── Universe Statistics ──")
    for k, v in report["universe_statistics"].items():
        print(f"  {k:<25}: {v}")

    print("\n── Clustering Quality ──")
    for k, v in report["clustering_quality"].items():
        try:
            print(f"  {k:<35}: {float(v):.4f}")
        except (TypeError, ValueError):
            print(f"  {k:<35}: {v}")

    print("\n── Portfolio Metrics ──")
    for universe, metrics in report["portfolio_metrics"].items():
        print(f"\n  [{universe}]")
        for k, v in metrics.items():
            try:
                print(f"    {k:<28}: {float(v):.4f}")
            except (TypeError, ValueError):
                print(f"    {k:<28}: {v}")
    print()
