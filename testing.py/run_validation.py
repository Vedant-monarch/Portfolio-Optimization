# Validation: true optimum (exact, full universe) vs Cluster + QAOA
#
# Usage:
#   python3 cluster_qpo/testing.py/run_validation.py            # K = 3, 4
#   python3 cluster_qpo/testing.py/run_validation.py 3 4 5     # custom K list
#
# K=5 over 99 assets is 71.5M exact evaluations (a few minutes).

import sys
import contextlib
import io

from cluster_qpo.data import SP100, load_returns
from cluster_qpo.validation import (
    validate_clustering_vs_optimum,
    print_validation_report
)

import os
METHOD = os.environ.get("METHOD", "centroid")
ks = [int(a) for a in sys.argv[1:]] or [3, 4]
print(f"Representative method: {METHOD}")

returns = load_returns(SP100, start="2019-01-01", end="2024-01-01", cache=True)
print(f"Universe: {returns.shape[1]} assets, {len(returns)} days\n")

all_tables = []
for k in ks:
    # Silence the verbose per-step QAOA printing; show only the validation report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = validate_clustering_vs_optimum(
            returns,
            budget=k,
            n_clusters=20,
            representative_method=METHOD,
            qaoa_kwargs=dict(p=2, shots=4096, maxiter=200),
            verbose=False
        )
    print_validation_report(result)
    all_tables.append((k, result))

print("=" * 70)
print("SUMMARY")
print("=" * 70)
for k, r in all_tables:
    qaoa = r["qaoa"]
    opt = r["optimum"]
    if qaoa is None:
        print(f"K={k}: QAOA returned no portfolio")
        continue
    row = r["table"].loc["Cluster + QAOA"]
    print(f"K={k}: QAOA recovers {row['% of optimum recovered']}% of optimum | "
          f"Sharpe {qaoa['sharpe']:.2f} vs optimum {opt['sharpe']:.2f} | "
          f"overlap {r['overlap_with_optimum']}/{k} | feasible={r['qaoa_feasible']}")
