import pandas as pd
from cluster_qpo.correlation import *

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

cov = covariance_matrix(returns)
corr = correlation_matrix(returns)
dist = correlation_to_distance(corr)

print(cov.shape)
print(corr.shape)
print(dist.shape)