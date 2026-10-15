# qpo/clustering/pipeline.py

import pandas as pd
from .correlation import (
    correlation_matrix,
    correlation_to_distance
)
from .hrp import (
    hierarchical_clustering,
    create_clusters,
    cluster_members
)
from .representatives import (
    representatives_by_centroid,
    representatives_by_sharpe,
    extract_representative_universe
)


def run_clustering_reduction(
    returns: pd.DataFrame,
    n_clusters: int = 20,
    method: str = "centroid",
    linkage_method: str = "ward"
) -> dict:
    """
    Full clustering and reduction pipeline.

    Combines correlation, hierarchical clustering,
    representative selection, and universe reduction
    into a single call.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows = timestamps, Columns = assets
    n_clusters : int
        Number of clusters to form
    method : str
        Representative selection method.
        Options: "centroid", "sharpe"
    linkage_method : str
        Linkage method for hierarchical clustering.
        Options: ward, complete, average, single

    Returns
    -------
    dict with keys:
        - "clusters"        : dict[int, list[str]]
        - "representatives" : pd.DataFrame
        - "reduced_returns" : pd.DataFrame
    """
    # Step 1 — Build correlation & distance matrix
    corr = correlation_matrix(returns)
    dist = correlation_to_distance(corr)

    # Step 2 — Hierarchical clustering
    linkage_matrix = hierarchical_clustering(
        dist,
        method=linkage_method
    )

    # Step 3 — Assign cluster labels
    cluster_labels = create_clusters(
        linkage_matrix,
        assets=returns.columns,
        n_clusters=n_clusters
    )

    # Step 4 — Get cluster members
    clusters = cluster_members(cluster_labels)

    # Step 5 — Select representatives
    if method == "centroid":
        reps = representatives_by_centroid(
            corr_matrix=corr,
            clusters=clusters
        )
    elif method == "sharpe":
        reps = representatives_by_sharpe(
            returns=returns,
            clusters=clusters
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose 'centroid' or 'sharpe'."
        )

    # Step 6 — Extract reduced returns
    reduced_returns = extract_representative_universe(
        returns,
        reps
    )

    return {
        "clusters": clusters,
        "representatives": reps,
        "reduced_returns": reduced_returns
    }
