import pandas as pd
from cluster_qpo.clustered_portfolio import ClusteredPortfolioProblem

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

problem = ClusteredPortfolioProblem(
    returns=returns,
    n_clusters=20,
    representative_method="centroid"
)

print("=== SUMMARY ===")
print(problem.summary())

print("\n=== REPR ===")
print(problem)

print("\n=== ASSET NAMES ===")
print(problem.get_asset_names())

print("\n=== mu shape ===")
print(problem.get_mu().shape)
print(problem.get_mu())

print("\n=== sigma shape ===")
print(problem.get_sigma().shape)

print("\n=== REPRESENTATIVES ===")
print(problem.get_representatives())

print("\n=== CLUSTER MAPPING ===")
for cid, members in problem.get_cluster_mapping().items():
    print(f"  Cluster {cid}: {members}")

print("\n=== QUBO READY INPUTS ===")
mu    = problem.get_mu()
sigma = problem.get_sigma()
print(f"  mu:    {mu.shape}")
print(f"  sigma: {sigma.shape}")
print("\n  → Ready to pass into build_portfolio_qubo(mu, sigma)")
