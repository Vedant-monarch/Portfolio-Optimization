# cluster_qpo/testing.py/run_real_data_experiment.py
#
# End-to-end experiment using real S&P 100 data.
#
# Steps:
#   1. Download S&P 100 prices (cached locally)
#   2. Compute returns
#   3. Run clustering pipeline
#   4. Evaluate cluster quality
#   5. Compare full vs reduced universe
#   6. Visualize clusters
#   7. Run benchmark report

import pandas as pd
from cluster_qpo.data import SP100, load_returns
from cluster_qpo.pipeline import run_clustering_reduction
from cluster_qpo.evaluation import evaluate_clustering
from cluster_qpo.backtest import compare_reduced_vs_full
from cluster_qpo.benchmark import (
    benchmark_clustering_pipeline,
    print_benchmark_report
)
from cluster_qpo.plots import (
    plot_dendrogram,
    plot_correlation_heatmap,
    plot_cluster_members
)
from cluster_qpo.correlation import correlation_matrix
from cluster_qpo.hrp import hierarchical_clustering, create_clusters

# ---------------------------------------------------------------
# Step 1 — Load Real Data
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1 — LOADING S&P 100 RETURNS")
print("=" * 60)

returns = load_returns(
    tickers=SP100,
    start="2019-01-01",
    end="2024-01-01",
    cache=True
)

print(f"  Universe : {returns.shape[1]} assets")
print(f"  Period   : {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"  Days     : {len(returns)}")
print()

# ---------------------------------------------------------------
# Step 2 — Run Clustering Pipeline
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 2 — RUNNING CLUSTERING PIPELINE")
print("=" * 60)

N_CLUSTERS = 20

result = run_clustering_reduction(
    returns=returns,
    n_clusters=N_CLUSTERS,
    method="centroid"
)

clusters      = result["clusters"]
reps          = result["representatives"]
reduced       = result["reduced_returns"]

print(f"  Original assets   : {returns.shape[1]}")
print(f"  Clusters formed   : {N_CLUSTERS}")
print(f"  Representatives   : {reduced.shape[1]}")
print()
print("Representatives:")
print(reps)
print()

# ---------------------------------------------------------------
# Step 3 — Evaluate Cluster Quality
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 3 — EVALUATING CLUSTER QUALITY")
print("=" * 60)

clustering_report = evaluate_clustering(
    returns=returns,
    clusters=clusters,
    representatives=reps,
    reduced_returns=reduced
)

print(clustering_report["summary"])
print()

# ---------------------------------------------------------------
# Step 4 — Information Loss Test
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 4 — INFORMATION LOSS TEST (Full vs Reduced)")
print("=" * 60)

comparison = compare_reduced_vs_full(
    returns=returns,
    n_clusters=N_CLUSTERS,
    representative_method="centroid"
)

print(comparison)
print()

# ---------------------------------------------------------------
# Step 5 — Benchmark Report
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 5 — BENCHMARK REPORT")
print("=" * 60)

report = benchmark_clustering_pipeline(
    full_returns=returns,
    reduced_returns=reduced,
    clustering_report=clustering_report
)

print_benchmark_report(report)

# ---------------------------------------------------------------
# Step 6 — Visualize Clusters
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 6 — VISUALIZING CLUSTERS")
print("=" * 60)

# Build linkage matrix for dendrogram
corr  = correlation_matrix(returns)
from cluster_qpo.correlation import correlation_to_distance
dist  = correlation_to_distance(corr)
lm    = hierarchical_clustering(dist, method="ward")
labels = create_clusters(lm, assets=returns.columns, n_clusters=N_CLUSTERS)

plot_dendrogram(
    linkage_matrix=lm,
    assets=list(returns.columns),
    title=f"S&P 100 — Hierarchical Clustering (Ward)",
    save_path="cluster_qpo/testing.py/dendrogram.png"
)

plot_correlation_heatmap(
    corr_matrix=corr,
    title="S&P 100 — Correlation Matrix (sorted by cluster)",
    cluster_labels=labels,
    save_path="cluster_qpo/testing.py/heatmap.png"
)

plot_cluster_members(
    cluster_labels=labels,
    representatives=reps,
    title="S&P 100 — Cluster Assignments (red = representative)",
    save_path="cluster_qpo/testing.py/cluster_members.png"
)

print("\nAll plots saved to cluster_qpo/testing.py/")
print("\nExperiment complete ✅")
