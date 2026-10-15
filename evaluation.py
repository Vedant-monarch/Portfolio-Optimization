# qpo/clustering/evaluation.py

import numpy as np
import pandas as pd


def intra_cluster_correlation(
    corr_matrix: pd.DataFrame,
    clusters: dict[int, list[str]]
) -> pd.DataFrame:
    """
    Compute average pairwise correlation
    within each cluster.

    Higher = assets inside cluster are similar.
    Good clustering → high intra-cluster correlation.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
    clusters : dict[int, list[str]]

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, intra_corr, size
    """
    results = []
    for cluster_id, assets in clusters.items():
        if len(assets) == 1:
            results.append({
                "cluster_id": cluster_id,
                "intra_corr": 1.0,
                "size": 1
            })
            continue

        sub_corr = corr_matrix.loc[assets, assets]

        # Extract upper triangle (exclude diagonal)
        mask = np.triu(np.ones(sub_corr.shape), k=1).astype(bool)
        pairwise = sub_corr.values[mask]

        results.append({
            "cluster_id": cluster_id,
            "intra_corr": float(np.mean(pairwise)),
            "size": len(assets)
        })

    return (
        pd.DataFrame(results)
        .set_index("cluster_id")
        .sort_index()
    )


def inter_cluster_correlation(
    corr_matrix: pd.DataFrame,
    representatives: pd.DataFrame
) -> float:
    """
    Compute average pairwise correlation
    between cluster representatives.

    Lower = clusters are more distinct from each other.
    Good clustering → low inter-cluster correlation.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
    representatives : pd.DataFrame
        Must have 'representative' column.

    Returns
    -------
    float
        Mean inter-cluster correlation
    """
    reps = representatives["representative"].tolist()

    if len(reps) == 1:
        return 0.0

    rep_corr = corr_matrix.loc[reps, reps]

    mask = np.triu(np.ones(rep_corr.shape), k=1).astype(bool)
    pairwise = rep_corr.values[mask]

    return float(np.mean(pairwise))


def representative_reconstruction_quality(
    returns: pd.DataFrame,
    clusters: dict[int, list[str]],
    representatives: pd.DataFrame
) -> pd.DataFrame:
    """
    Measure how well each representative
    captures its cluster members.

    Metric: average correlation between
    representative and each cluster member.

    Parameters
    ----------
    returns : pd.DataFrame
    clusters : dict[int, list[str]]
    representatives : pd.DataFrame
        Must have 'representative' column.

    Returns
    -------
    pd.DataFrame
        Columns: cluster_id, representative,
                 mean_rep_corr, min_rep_corr
    """
    corr_matrix = returns.corr()
    results = []

    for cluster_id, assets in clusters.items():
        rep = representatives.loc[cluster_id, "representative"]

        if len(assets) == 1:
            results.append({
                "cluster_id": cluster_id,
                "representative": rep,
                "mean_rep_corr": 1.0,
                "min_rep_corr": 1.0
            })
            continue

        members = [a for a in assets if a != rep]
        correlations = [
            corr_matrix.loc[rep, member]
            for member in members
        ]

        results.append({
            "cluster_id": cluster_id,
            "representative": rep,
            "mean_rep_corr": float(np.mean(correlations)),
            "min_rep_corr": float(np.min(correlations))
        })

    return (
        pd.DataFrame(results)
        .set_index("cluster_id")
        .sort_index()
    )


def variance_retained(
    returns: pd.DataFrame,
    reduced_returns: pd.DataFrame
) -> float:
    """
    Compute fraction of total variance
    retained by the reduced universe.

    Formula:
        variance_retained = sum(var(reduced)) / sum(var(full))

    Parameters
    ----------
    returns : pd.DataFrame
        Full universe returns
    reduced_returns : pd.DataFrame
        Representative universe returns

    Returns
    -------
    float
        Fraction between 0 and 1
    """
    total_var = returns.var().sum()
    reduced_var = reduced_returns.var().sum()

    if total_var == 0:
        return 0.0

    return float(reduced_var / total_var)


def average_cluster_size(
    clusters: dict[int, list[str]]
) -> float:
    """
    Compute average number of assets per cluster.

    Parameters
    ----------
    clusters : dict[int, list[str]]

    Returns
    -------
    float
    """
    sizes = [len(assets) for assets in clusters.values()]
    return float(np.mean(sizes))


def evaluate_clustering(
    returns: pd.DataFrame,
    clusters: dict[int, list[str]],
    representatives: pd.DataFrame,
    reduced_returns: pd.DataFrame
) -> dict:
    """
    Run full evaluation suite.

    Answers: "Did clustering destroy information?"

    Parameters
    ----------
    returns : pd.DataFrame
        Full universe returns
    clusters : dict[int, list[str]]
    representatives : pd.DataFrame
    reduced_returns : pd.DataFrame

    Returns
    -------
    dict with keys:
        - "intra_cluster_correlation"  : pd.DataFrame
        - "inter_cluster_correlation"  : float
        - "reconstruction_quality"     : pd.DataFrame
        - "variance_retained"          : float
        - "average_cluster_size"       : float
        - "summary"                    : pd.DataFrame
    """
    corr_matrix = returns.corr()

    intra = intra_cluster_correlation(corr_matrix, clusters)
    inter = inter_cluster_correlation(corr_matrix, representatives)
    recon = representative_reconstruction_quality(
        returns, clusters, representatives
    )
    var_ret = variance_retained(returns, reduced_returns)
    avg_size = average_cluster_size(clusters)

    summary = pd.DataFrame({
        "metric": [
            "Mean Intra-Cluster Correlation",
            "Inter-Cluster Correlation",
            "Mean Representative Correlation",
            "Variance Retained",
            "Average Cluster Size",
            "Num Clusters",
            "Original Universe Size",
            "Reduced Universe Size"
        ],
        "value": [
            round(intra["intra_corr"].mean(), 4),
            round(inter, 4),
            round(recon["mean_rep_corr"].mean(), 4),
            round(var_ret, 4),
            round(avg_size, 2),
            len(clusters),
            len(returns.columns),
            len(reduced_returns.columns)
        ]
    }).set_index("metric")

    return {
        "intra_cluster_correlation": intra,
        "inter_cluster_correlation": inter,
        "reconstruction_quality": recon,
        "variance_retained": var_ret,
        "average_cluster_size": avg_size,
        "summary": summary
    }
