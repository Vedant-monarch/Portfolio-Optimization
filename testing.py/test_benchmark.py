import pandas as pd
from cluster_qpo.pipeline import run_clustering_reduction
from cluster_qpo.evaluation import evaluate_clustering
from cluster_qpo.benchmark import (
    benchmark_clustering_pipeline,
    print_benchmark_report
)

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

result = run_clustering_reduction(
    returns=returns,
    n_clusters=10,
    method="centroid"
)

clustering_report = evaluate_clustering(
    returns=returns,
    clusters=result["clusters"],
    representatives=result["representatives"],
    reduced_returns=result["reduced_returns"]
)

report = benchmark_clustering_pipeline(
    full_returns=returns,
    reduced_returns=result["reduced_returns"],
    clustering_report=clustering_report
)

print_benchmark_report(report)
