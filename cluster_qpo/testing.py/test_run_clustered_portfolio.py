import pandas as pd
from cluster_qpo.run_clustered_portfolio import (
    run_clustered_portfolio_experiment
)

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

result = run_clustered_portfolio_experiment(
    returns=returns,
    n_clusters=10,
    cardinality=3,
    representative_method="centroid",
    p=1,
    shots=2048,
    optimizer="cobyla",
    maxiter=100,
    verbose=False
)

print("\n=== FINAL RESULT ===")
print(f"Selected Assets  : {result['selected_assets']}")
print(f"Selected Clusters: {result['selected_clusters']}")
print(f"Feasible         : {result['solver_result'].solution.feasible}")
print(f"Expected Return  : {result['solver_result'].solution.expected_return:.6f}")
print(f"Volatility       : {result['solver_result'].solution.volatility:.6f}")
print(f"Sharpe           : {result['solver_result'].solution.sharpe_like:.4f}")
