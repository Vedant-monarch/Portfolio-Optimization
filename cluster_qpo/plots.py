# qpo/clustering/plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(
    linkage_matrix,
    assets: list[str],
    title: str = "Asset Hierarchical Clustering",
    figsize: tuple = (14, 6),
    save_path: str = None
) -> None:
    """
    Plot hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix : ndarray
    assets : list[str]
    title : str
    figsize : tuple
    save_path : str  optional path to save PNG
    """
    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=assets,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Assets", fontsize=11)
    ax.set_ylabel("Distance", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved dendrogram → {save_path}")

    plt.show()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    figsize: tuple = (12, 10),
    cluster_labels: pd.Series = None,
    save_path: str = None
) -> None:
    """
    Plot correlation heatmap, optionally
    sorted by cluster assignment.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
    title : str
    figsize : tuple
    cluster_labels : pd.Series
        If provided, assets are sorted by cluster.
        index = asset, value = cluster_id
    save_path : str
    """
    # Sort by cluster if labels provided
    if cluster_labels is not None:
        sorted_assets = (
            cluster_labels
            .sort_values()
            .index
            .tolist()
        )
        corr_sorted = corr_matrix.loc[sorted_assets, sorted_assets]
    else:
        corr_sorted = corr_matrix

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_sorted,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        xticklabels=True,
        yticklabels=True
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=7, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved heatmap → {save_path}")

    plt.show()


def plot_cluster_members(
    cluster_labels: pd.Series,
    representatives: pd.DataFrame = None,
    title: str = "Cluster Assignments",
    figsize: tuple = (14, 6),
    save_path: str = None
) -> None:
    """
    Bar chart showing assets grouped by cluster.
    Representatives are highlighted in a different colour.

    Parameters
    ----------
    cluster_labels : pd.Series
        index = asset, value = cluster_id
    representatives : pd.DataFrame
        Must have 'representative' column
    title : str
    figsize : tuple
    save_path : str
    """
    df = cluster_labels.reset_index()
    df.columns = ["asset", "cluster_id"]
    df = df.sort_values("cluster_id")

    # Identify representatives
    rep_set = set()
    if representatives is not None:
        rep_set = set(representatives["representative"].tolist())

    colors = [
        "#e74c3c" if asset in rep_set else "#3498db"
        for asset in df["asset"]
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        df["asset"],
        df["cluster_id"],
        color=colors,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Asset", fontsize=11)
    ax.set_ylabel("Cluster ID", fontsize=11)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Representative"),
        Patch(facecolor="#3498db", label="Cluster Member")
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plots] Saved cluster plot → {save_path}")

    plt.show()
