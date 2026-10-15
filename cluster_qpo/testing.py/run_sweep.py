# Sweep representative method x n_clusters x K against the reference optimum.
#
# Usage:
#   python3 -u cluster_qpo/testing.py/run_sweep.py sp100
#   python3 -u cluster_qpo/testing.py/run_sweep.py sp500
#
# No QAOA here (see validation.sweep_representatives docstring).

import sys
import pandas as pd

from cluster_qpo.data import SP100, SP500, load_returns
from cluster_qpo.validation import sweep_representatives

universe = (sys.argv[1] if len(sys.argv) > 1 else "sp100").lower()
tickers = {"sp100": SP100, "sp500": SP500}[universe]
cluster_counts = (20, 30, 40) if universe == "sp100" else (20, 40, 60, 80)

returns = load_returns(tickers, start="2019-01-01", end="2024-01-01", cache=True)
print(f"\nUniverse {universe}: {returns.shape[1]} assets, {len(returns)} days\n")

df = sweep_representatives(
    returns,
    budgets=(3, 4, 5),
    methods=("centroid", "sharpe"),
    cluster_counts=cluster_counts,
)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print("\n" + "=" * 100)
print(f"SWEEP RESULTS - {universe.upper()}")
print("=" * 100)
print(df.to_string(index=False))

print("\nMean % of reference recovered, by method x n_clusters:")
print(df.pivot_table(index="method", columns="n_clusters",
                     values="% of reference recovered", aggfunc="mean").round(1))

out = f"cluster_qpo/testing.py/sweep_{universe}.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")
