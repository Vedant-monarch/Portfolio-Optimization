# qpo/clustering/representatives.py

import numpy as np
import pandas as pd
from .correlation import sharpe_like_scores


def centroid_representative(
    corr_matrix: pd.DataFrame,
    cluster_assets: list[str]
) -> str:
    """
    Select asset closest to cluster centroid.

    The chosen asset is the one whose correlation
    profile is closest to the average profile
    within the cluster.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
    cluster_assets : list[str]

    Returns
    -------
    str
        Representative asset ticker
    """
    if len(cluster_assets) == 1:
        return cluster_assets[0]

    cluster_corr = corr_matrix.loc[
        cluster_assets,
        cluster_assets
    ]

    centroid = cluster_corr.mean(axis=0)

    distances = {}
    for asset in cluster_assets:
        distances[asset] = np.linalg.norm(
            cluster_corr.loc[asset].values
            - centroid.values
        )

    representative = min(
        distances,
        key=distances.get
    )

    return representative


def sharpe_representative(
    returns: pd.DataFrame,
    cluster_assets: list[str]
) -> str:
    """
    Select highest Sharpe-like asset
    inside a cluster.

    Parameters
    ----------
    returns : pd.DataFrame
    cluster_assets : list[str]

    Returns
    -------
    str
    """
    cluster_returns = returns[cluster_assets]
    scores = sharpe_like_scores(cluster_returns)
    return scores.idxmax()


def representatives_by_centroid(
    corr_matrix: pd.DataFrame,
    clusters: dict[int, list[str]]
) -> pd.DataFrame:
    """
    Get one representative asset
    per cluster using centroid method.

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, representative, cluster_size
    """
    results = []
    for cluster_id, assets in clusters.items():
        representative = centroid_representative(
            corr_matrix=corr_matrix,
            cluster_assets=assets
        )
        results.append({
            "cluster_id": cluster_id,
            "representative": representative,
            "cluster_size": len(assets)
        })

    return (
        pd.DataFrame(results)
        .set_index("cluster_id")
        .sort_index()
    )


def representatives_by_sharpe(
    returns: pd.DataFrame,
    clusters: dict[int, list[str]]
) -> pd.DataFrame:
    """
    Get one representative asset
    per cluster using Sharpe-like score method.

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, representative, cluster_size
    """
    results = []
    for cluster_id, assets in clusters.items():
        representative = sharpe_representative(
            returns=returns,
            cluster_assets=assets
        )
        results.append({
            "cluster_id": cluster_id,
            "representative": representative,
            "cluster_size": len(assets)
        })

    return (
        pd.DataFrame(results)
        .set_index("cluster_id")
        .sort_index()
    )


def extract_representative_universe(
    returns: pd.DataFrame,
    reps: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter returns DataFrame to only include
    representative assets from each cluster.

    Parameters
    ----------
    returns : pd.DataFrame
        Full returns DataFrame (rows=dates, cols=assets)
    reps : pd.DataFrame
        Output of representatives_by_centroid or
        representatives_by_sharpe. Must have a
        'representative' column.

    Returns
    -------
    pd.DataFrame
        Reduced returns with only representative assets
    """
    representative_assets = reps["representative"].tolist()
    return returns[representative_assets]
