import pandas as pd
from cluster_qpo.reduction import build_reduced_portfolio_inputs

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

result = build_reduced_portfolio_inputs(
    returns=returns,
    n_clusters=20,
    representative_method="centroid"
)

mu = result["mu"]
sigma = result["sigma"]

print("mu shape:", mu.shape)
print("sigma shape:", sigma.shape)

print("\nmu:")
print(mu)

print("\nsigma:")
print(sigma)

print("\nRepresentatives:")
print(result["representatives"])
