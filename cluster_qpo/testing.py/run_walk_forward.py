# Walk-forward (train/test) validation.
#
# Usage:
#   python3 -u cluster_qpo/testing.py/run_walk_forward.py sp500
#   python3 -u cluster_qpo/testing.py/run_walk_forward.py sp100
#
# Train 2 years -> pick portfolios -> score on the next unseen year; roll forward.

import sys
import pandas as pd

from cluster_qpo.data import SP100, SP500, load_returns
from cluster_qpo.validation import walk_forward_validation

universe = (sys.argv[1] if len(sys.argv) > 1 else "sp500").lower()
tickers = {"sp100": SP100, "sp500": SP500}[universe]

returns = load_returns(tickers, start="2019-01-01", end="2024-01-01", cache=True)
print(f"\nUniverse {universe}: {returns.shape[1]} assets, {len(returns)} days\n", flush=True)

df = walk_forward_validation(
    returns,
    budgets=(3, 5),
    cluster_counts=(20, 40, 60),
    method="sharpe",
    train_days=504,
    test_days=252,
)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
cols = ["fold", "test_window", "K", "n_clusters", "cluster_oos_sharpe",
        "ref_oos_sharpe", "random_mean_sharpe", "universe_ew_sharpe"]
print("\n" + "=" * 100)
print(f"WALK-FORWARD OUT-OF-SAMPLE SHARPE - {universe.upper()}")
print("=" * 100)
print(df[cols].to_string(index=False))

print("\nMean out-of-sample Sharpe across folds:")
summary = df.groupby(["K", "n_clusters"])[
    ["cluster_oos_sharpe", "ref_oos_sharpe", "random_mean_sharpe", "universe_ew_sharpe"]
].mean().round(3)
print(summary)

out = f"cluster_qpo/testing.py/walk_forward_{universe}.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")
