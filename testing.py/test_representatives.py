import pandas as pd
from cluster_qpo.correlation import (
    correlation_matrix,
    correlation_to_distance
)
from cluster_qpo.hrp import (
    hierarchical_clustering,
    create_clusters,
    cluster_members
)
from cluster_qpo.representatives import (
    representatives_by_centroid
)

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

corr = correlation_matrix(returns)
dist = correlation_to_distance(corr)

linkage_matrix = hierarchical_clustering(
    dist,
    method="ward"
)

cluster_labels = create_clusters(
    linkage_matrix,
    assets=returns.columns,
    n_clusters=20
)

clusters = cluster_members(cluster_labels)

reps = representatives_by_centroid(
    corr_matrix=corr,
    clusters=clusters
)

print(reps)
