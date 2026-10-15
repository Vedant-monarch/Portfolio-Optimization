# qpo/clustering/hrp.py

import pandas as pd
from scipy.cluster.hierarchy import (
    linkage,
    fcluster
)
from scipy.spatial.distance import squareform


def hierarchical_clustering(
    distance_matrix: pd.DataFrame,
    method: str = "ward"
):
    """
    Build hierarchical clustering linkage matrix.

    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Pairwise asset distance matrix.
    method : str
        ward, complete, average, single

    Returns
    -------
    ndarray
        Linkage matrix
    """
    condensed_distance = squareform(
        distance_matrix.values,
        checks=False
    )
    linkage_matrix = linkage(
        condensed_distance,
        method=method
    )
    return linkage_matrix


def create_clusters(
    linkage_matrix,
    assets,
    n_clusters: int = 20
):
    """
    Convert linkage matrix into cluster labels.

    Parameters
    ----------
    linkage_matrix : ndarray
    assets : list[str]
    n_clusters : int

    Returns
    -------
    pd.Series
        index = asset names
        value = cluster id
    """
    labels = fcluster(
        linkage_matrix,
        t=n_clusters,
        criterion="maxclust"
    )
    return pd.Series(
        labels,
        index=assets,
        name="cluster"
    )


def cluster_members(
    cluster_labels: pd.Series
):
    """
    Convert asset->cluster mapping
    into cluster->assets mapping.

    Parameters
    ----------
    cluster_labels : pd.Series

    Returns
    -------
    dict[int, list[str]]
    """
    clusters = {}
    for asset, cluster_id in cluster_labels.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(asset)
    return clusters


def cluster_summary(
    cluster_labels: pd.Series
) -> pd.DataFrame:
    """
    Create summary table.

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, size, assets
    """
    members = cluster_members(cluster_labels)
    summary = pd.DataFrame([
        {
            "cluster_id": cluster_id,
            "size": len(assets),
            "assets": ", ".join(assets)
        }
        for cluster_id, assets in sorted(members.items())
    ])
    return summary.set_index("cluster_id")
