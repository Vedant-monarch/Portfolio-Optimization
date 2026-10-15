import pandas as pd
from cluster_qpo.pipeline import run_clustering_reduction
from cluster_qpo.evaluation import evaluate_clustering

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

result = run_clustering_reduction(
    returns,
    n_clusters=20,
    method="centroid"
)

evaluation = evaluate_clustering(
    returns=returns,
    clusters=result["clusters"],
    representatives=result["representatives"],
    reduced_returns=result["reduced_returns"]
)

print("=== SUMMARY ===")
print(evaluation["summary"])

print("\n=== INTRA-CLUSTER CORRELATION ===")
print(evaluation["intra_cluster_correlation"])

print("\n=== INTER-CLUSTER CORRELATION ===")
print(round(evaluation["inter_cluster_correlation"], 4))

print("\n=== RECONSTRUCTION QUALITY ===")
print(evaluation["reconstruction_quality"])

print("\n=== VARIANCE RETAINED ===")
print(round(evaluation["variance_retained"], 4))

print("\n=== AVERAGE CLUSTER SIZE ===")
print(round(evaluation["average_cluster_size"], 2))
