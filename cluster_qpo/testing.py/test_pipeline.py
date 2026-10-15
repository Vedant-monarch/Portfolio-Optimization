import pandas as pd
from cluster_qpo.pipeline import run_clustering_reduction

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

result = run_clustering_reduction(
    returns,
    n_clusters=20,
    method="centroid"
)

print("Clusters:")
for cid, members in result["clusters"].items():
    print(f"  Cluster {cid}: {members}")

print("\nRepresentatives:")
print(result["representatives"])

print("\nReduced Returns Shape:")
print(result["reduced_returns"].shape)
