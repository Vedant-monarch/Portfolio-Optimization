# qpo/cluster_qpo/selection_analysis.py

import numpy as np
import pandas as pd


def cluster_coverage(
    selected_assets: list[str],
    clusters: dict[int, list[str]]
) -> dict:
    """
    How much of the original universe is
    represented by the selected clusters?

    Parameters
    ----------
    selected_assets : list[str]
        Assets chosen by QAOA
    clusters : dict[int, list[str]]
        Full cluster → assets mapping

    Returns
    -------
    dict with keys:
        - selected_clusters     : list[int]
        - n_selected_clusters   : int
        - total_clusters        : int
        - assets_represented    : list[str]
        - n_assets_represented  : int
        - total_assets          : int
        - coverage_ratio        : float
    """
    # Map asset → cluster_id
    asset_to_cluster = {
        asset: cid
        for cid, assets in clusters.items()
        for asset in assets
    }

    selected_cluster_ids = list({
        asset_to_cluster[a]
        for a in selected_assets
        if a in asset_to_cluster
    })

    # All assets covered by selected clusters
    assets_represented = [
        asset
        for cid in selected_cluster_ids
        for asset in clusters[cid]
    ]

    total_assets = sum(len(v) for v in clusters.values())

    return {
        "selected_clusters": sorted(selected_cluster_ids),
        "n_selected_clusters": len(selected_cluster_ids),
        "total_clusters": len(clusters),
        "assets_represented": assets_represented,
        "n_assets_represented": len(assets_represented),
        "total_assets": total_assets,
        "coverage_ratio": round(
            len(assets_represented) / total_assets, 4
        ) if total_assets > 0 else 0.0
    }


def average_cluster_size_selected(
    selected_assets: list[str],
    clusters: dict[int, list[str]]
) -> float:
    """
    Average size of the clusters that
    were selected by QAOA.

    Parameters
    ----------
    selected_assets : list[str]
    clusters : dict[int, list[str]]

    Returns
    -------
    float
    """
    asset_to_cluster = {
        asset: cid
        for cid, assets in clusters.items()
        for asset in assets
    }

    selected_cluster_ids = {
        asset_to_cluster[a]
        for a in selected_assets
        if a in asset_to_cluster
    }

    if not selected_cluster_ids:
        return 0.0

    sizes = [len(clusters[cid]) for cid in selected_cluster_ids]
    return round(float(np.mean(sizes)), 4)


def correlation_diversity(
    selected_assets: list[str],
    returns: pd.DataFrame
) -> dict:
    """
    Measure how diverse the selected
    assets are in terms of correlation.

    Low mean correlation = high diversity = good.

    Parameters
    ----------
    selected_assets : list[str]
    returns : pd.DataFrame

    Returns
    -------
    dict with keys:
        - mean_pairwise_corr : float
        - min_pairwise_corr  : float
        - max_pairwise_corr  : float
        - diversity_score    : float  (1 - mean_corr)
    """
    available = [a for a in selected_assets if a in returns.columns]

    if len(available) < 2:
        return {
            "mean_pairwise_corr": 1.0,
            "min_pairwise_corr": 1.0,
            "max_pairwise_corr": 1.0,
            "diversity_score": 0.0
        }

    corr = returns[available].corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairwise = corr.values[mask]

    mean_corr = float(np.mean(pairwise))

    return {
        "mean_pairwise_corr": round(mean_corr, 4),
        "min_pairwise_corr": round(float(np.min(pairwise)), 4),
        "max_pairwise_corr": round(float(np.max(pairwise)), 4),
        "diversity_score": round(1.0 - mean_corr, 4)
    }


def why_selected(
    asset: str,
    clusters: dict[int, list[str]],
    representatives: pd.DataFrame,
    returns: pd.DataFrame
) -> dict:
    """
    Explain why a given asset was selected
    as a cluster representative.

    Answers: "QAOA selected NVDA — why NVDA?"

    Parameters
    ----------
    asset : str
    clusters : dict[int, list[str]]
    representatives : pd.DataFrame
        Must have 'representative' column
    returns : pd.DataFrame

    Returns
    -------
    dict with keys:
        - asset               : str
        - cluster_id          : int
        - cluster_members     : list[str]
        - cluster_size        : int
        - mean_return         : float
        - volatility          : float
        - sharpe_like         : float
        - mean_intra_corr     : float  (rep vs cluster members)
        - selection_reason    : str
    """
    # Find cluster
    asset_to_cluster = {
        a: cid
        for cid, assets in clusters.items()
        for a in assets
    }

    if asset not in asset_to_cluster:
        return {"asset": asset, "error": "not found in clusters"}

    cluster_id = asset_to_cluster[asset]
    members = clusters[cluster_id]

    # Return stats
    if asset in returns.columns:
        r = returns[asset]
        mean_ret = float(r.mean())
        vol = float(r.std())
        sharpe = mean_ret / vol if vol > 0 else 0.0
    else:
        mean_ret = vol = sharpe = float("nan")

    # Intra-cluster correlation (rep vs members)
    others = [m for m in members if m != asset]
    if others and asset in returns.columns:
        corr_matrix = returns[members].corr()
        corrs = [corr_matrix.loc[asset, m] for m in others]
        mean_intra = round(float(np.mean(corrs)), 4)
    else:
        mean_intra = 1.0

    # Selection reason
    if len(members) == 1:
        reason = "Only asset in cluster — selected by default."
    elif mean_intra >= 0.8:
        reason = (
            f"Strong centroid representative "
            f"(mean intra-cluster corr = {mean_intra:.2f}). "
            f"Closely tracks all {len(members)} assets in cluster {cluster_id}."
        )
    else:
        reason = (
            f"Best centroid representative among {len(members)} assets "
            f"in cluster {cluster_id} "
            f"(mean intra-cluster corr = {mean_intra:.2f})."
        )

    return {
        "asset": asset,
        "cluster_id": cluster_id,
        "cluster_members": members,
        "cluster_size": len(members),
        "mean_return": round(mean_ret, 6),
        "volatility": round(vol, 6),
        "sharpe_like": round(sharpe, 4),
        "mean_intra_corr": mean_intra,
        "selection_reason": reason
    }


def selection_analysis(
    selected_assets: list[str],
    clusters: dict[int, list[str]],
    representatives: pd.DataFrame,
    returns: pd.DataFrame
) -> dict:
    """
    Full selection analysis suite.

    Answers:
        - Why was each asset selected?
        - How much of the original universe is covered?
        - How diverse is the selection?

    Parameters
    ----------
    selected_assets : list[str]
    clusters : dict[int, list[str]]
    representatives : pd.DataFrame
    returns : pd.DataFrame

    Returns
    -------
    dict with keys:
        - "coverage"               : dict
        - "avg_selected_cluster_size" : float
        - "correlation_diversity"  : dict
        - "asset_explanations"     : list[dict]
        - "summary"                : pd.DataFrame
    """
    coverage = cluster_coverage(selected_assets, clusters)
    avg_size = average_cluster_size_selected(selected_assets, clusters)
    diversity = correlation_diversity(selected_assets, returns)

    explanations = [
        why_selected(asset, clusters, representatives, returns)
        for asset in selected_assets
    ]

    summary = pd.DataFrame([
        {
            "asset": e["asset"],
            "cluster_id": e.get("cluster_id", -1),
            "cluster_size": e.get("cluster_size", 0),
            "mean_return": e.get("mean_return", float("nan")),
            "volatility": e.get("volatility", float("nan")),
            "sharpe_like": e.get("sharpe_like", float("nan")),
            "mean_intra_corr": e.get("mean_intra_corr", float("nan")),
        }
        for e in explanations
    ]).set_index("asset")

    return {
        "coverage": coverage,
        "avg_selected_cluster_size": avg_size,
        "correlation_diversity": diversity,
        "asset_explanations": explanations,
        "summary": summary
    }


def print_selection_report(analysis: dict) -> None:
    """
    Human-readable selection analysis output.
    """
    print("\n")
    print("=" * 60)
    print("SELECTION ANALYSIS REPORT")
    print("=" * 60)

    print("\n── Cluster Coverage ──")
    cov = analysis["coverage"]
    print(f"  Selected Clusters    : {cov['selected_clusters']}")
    print(f"  Clusters Selected    : {cov['n_selected_clusters']} / {cov['total_clusters']}")
    print(f"  Assets Represented   : {cov['n_assets_represented']} / {cov['total_assets']}")
    print(f"  Coverage Ratio       : {cov['coverage_ratio']:.2%}")

    print("\n── Diversity ──")
    div = analysis["correlation_diversity"]
    print(f"  Mean Pairwise Corr   : {div['mean_pairwise_corr']:.4f}")
    print(f"  Min Pairwise Corr    : {div['min_pairwise_corr']:.4f}")
    print(f"  Max Pairwise Corr    : {div['max_pairwise_corr']:.4f}")
    print(f"  Diversity Score      : {div['diversity_score']:.4f}")
    print(f"  Avg Selected Cluster : {analysis['avg_selected_cluster_size']:.2f} assets")

    print("\n── Asset Explanations ──")
    for e in analysis["asset_explanations"]:
        print(f"\n  {e['asset']}  (Cluster {e.get('cluster_id', '?')})")
        print(f"    Members    : {e.get('cluster_members', [])}")
        print(f"    Return     : {e.get('mean_return', 0):.6f}")
        print(f"    Volatility : {e.get('volatility', 0):.6f}")
        print(f"    Sharpe     : {e.get('sharpe_like', 0):.4f}")
        print(f"    Reason     : {e.get('selection_reason', '')}")
    print()
