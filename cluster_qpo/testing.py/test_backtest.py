import pandas as pd
from cluster_qpo.backtest import (
    portfolio_metrics,
    backtest_representative_universe,
    compare_reduced_vs_full
)

returns = pd.read_csv(
    "cluster_qpo/data/returns.csv",
    index_col=0
)

# --- Full Universe Metrics ---
print("=== FULL UNIVERSE METRICS ===")
full_metrics = portfolio_metrics(returns)
for k, v in full_metrics.items():
    print(f"  {k}: {v}")

# --- Reduced Universe Metrics ---
print("\n=== REDUCED UNIVERSE METRICS ===")
reduced = backtest_representative_universe(
    returns=returns,
    n_clusters=20,
    representative_method="centroid"
)
for k, v in reduced["metrics"].items():
    print(f"  {k}: {v}")

# --- Side-by-Side Comparison ---
print("\n=== FULL vs REDUCED COMPARISON ===")
comparison = compare_reduced_vs_full(
    returns=returns,
    n_clusters=20,
    representative_method="centroid"
)
print(comparison)
