# Many-fold rolling walk-forward test (6-month test windows).
#
# Usage:
#   python3 -u cluster_qpo/testing.py/run_rolling_walk_forward.py sp500 sharpe
#   python3 -u cluster_qpo/testing.py/run_rolling_walk_forward.py sp500 centroid
#
# Each fold: train on 504 days (2y), test on the next 126 days (6 months),
# then roll forward by 126 days. Test windows do not overlap.
# Longer history (2014-2024) gives many more folds than the 2019-2024 run.

import sys
import numpy as np
import pandas as pd
from scipy import stats

from cluster_qpo.data import SP100, SP500, load_returns
from cluster_qpo.validation import walk_forward_validation

universe = (sys.argv[1] if len(sys.argv) > 1 else "sp500").lower()
method = (sys.argv[2] if len(sys.argv) > 2 else "sharpe").lower()
tickers = {"sp100": SP100, "sp500": SP500}[universe]

returns = load_returns(tickers, start="2014-01-01", end="2024-01-01", cache=True)
print(f"\nUniverse {universe}: {returns.shape[1]} assets, {len(returns)} days "
      f"({returns.index[0].date()} -> {returns.index[-1].date()})", flush=True)
print(f"Representative method: {method}\n", flush=True)

df = walk_forward_validation(
    returns,
    budgets=(3, 5),
    cluster_counts=(20, 40, 60),
    method=method,
    train_days=504,
    test_days=126,
    step_days=126,
)

n_folds = df["fold"].nunique()
print(f"\nFolds completed: {n_folds}")

# ---------------------------------------------------------------
# Aggregate statistics across folds
# ---------------------------------------------------------------
rows = []
for (K, nc), g in df.groupby(["K", "n_clusters"]):
    c, ref = g["cluster_oos_sharpe"], g["ref_oos_sharpe"]
    rnd, uni = g["random_mean_sharpe"], g["universe_ew_sharpe"]

    def paired(a, b):
        d = (a - b).values
        if len(d) < 3 or d.std(ddof=1) == 0:
            return np.nan
        return stats.ttest_1samp(d, 0.0).pvalue

    rows.append({
        "K": K,
        "n_clusters": nc,
        "folds": len(g),
        "cluster_mean": c.mean(),
        "cluster_std": c.std(),
        "ref_mean": ref.mean(),
        "random_mean": rnd.mean(),
        "universe_mean": uni.mean(),
        "win_vs_random_%": (c > rnd).mean() * 100,
        "win_vs_universe_%": (c > uni).mean() * 100,
        "win_vs_ref_%": (c > ref).mean() * 100,
        "p(cluster>random)": paired(c, rnd),
        "p(cluster>universe)": paired(c, uni),
    })
agg = pd.DataFrame(rows).round(3)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n" + "=" * 100)
print(f"ROLLING WALK-FORWARD ({n_folds} folds x 126-day test) - "
      f"{universe.upper()} / {method}")
print("=" * 100)
print(agg.to_string(index=False))
print("\nwin_% = share of folds where the cluster portfolio had higher "
      "out-of-sample Sharpe.")
print("p = two-sided paired t-test on per-fold Sharpe difference "
      "(small = unlikely to be chance).")

out = f"cluster_qpo/testing.py/rolling_wf_{universe}_{method}"
df.to_csv(out + ".csv", index=False)
agg.to_csv(out + "_summary.csv", index=False)
print(f"\nSaved {out}.csv and {out}_summary.csv")
