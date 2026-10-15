# Clustering for Quantum Portfolio Optimization: One-Page Summary

## The idea
QAOA can only handle about 20 assets (one qubit each), but real portfolios have hundreds.
We compress the universe first:

```
Many stocks -> correlation -> distance -> hierarchical (Ward) clustering
            -> 1 representative per cluster -> ~20 assets -> QUBO -> QAOA
```

The module (`cluster_qpo/`) is pure Python math (pandas / numpy / scipy). Only the final solver
step touches quantum code, so the clustering is portable across backends.

## Data
Daily adjusted closes from Yahoo Finance (`yfinance`), converted to simple daily returns.

| Dataset | Period | Size | Used for |
|---|---|---|---|
| S&P 100 | 2019-2024 | 99 stocks, 1,257 days | Main experiments, QAOA validation |
| Large universe | 2019-2024 | 380 stocks, 1,158 days | Sweep, 2-fold walk-forward |
| Large universe | 2014-2024 | 368 stocks, 2,314 days | 14-fold rolling walk-forward |

## Results

**1. Clustering produces sensible groups (S&P 100, 99 -> 20 assets)**

| Metric | Value |
|---|---|
| Intra-cluster correlation | 0.64 |
| Inter-cluster correlation | 0.43 |
| Representative-to-cluster correlation | 0.68 |
| Sharpe, equal-weight, all 99 vs all 20 | 0.997 vs 0.959 (-3.9%) |

Energy, banks, semiconductors and utilities each grouped together.

**2. QAOA solves the reduced problem exactly (S&P 100, with `sharpe` representatives)**

| K | True optimum (exact search over all 99) | Cluster + QAOA | Optimum recovered |
|---|---|---|---|
| 3 | COST, LLY, NVDA | NVDA, LLY, COST | 100% |
| 4 | COST, LLY, NVDA, WM | NVDA, LLY, COST, WM | 100% |

**3. The representative rule matters (mean % of reference optimum recovered)**

| Universe | Method | 20 clusters | 40 clusters |
|---|---|---|---|
| S&P 100 | centroid | 52.9% | 93.6% |
| S&P 100 | sharpe | 100% | 100% |
| Large (380) | centroid | 22.2% | 64.4% |
| Large (380) | sharpe | 90.7% | 100% |

`centroid` picks the most typical stock per cluster and misses the standout ones
(NVDA, LLY, TSLA). `sharpe` keeps them.

**4. Out-of-sample, the edge disappears (14 rolling folds, 6-month test windows)**

| | Cluster portfolio (`sharpe`) | Random K-portfolio | Equal-weight universe |
|---|---|---|---|
| Mean out-of-sample Sharpe | 1.00 - 1.07 | 0.88 - 0.95 | 1.27 |

- The cluster portfolio beat random picks in about **50%** of folds, which is a coin flip.
- Paired t-test p-values against random: **0.55 - 0.81**, so nothing is significant.
- `sharpe` vs `centroid` makes no reliable difference out-of-sample.

## Main findings
1. **Clustering preserves structure.** The groups are economically meaningful, and the compressed
   universe behaves like the full one in an equal-weight portfolio.
2. **QAOA is not the weak link.** It matched the exact best portfolio among the representatives
   every time we tested it.
3. **The representative rule decides in-sample quality.** `sharpe` recovers the optimum, `centroid`
   loses roughly half of it.
4. **That in-sample advantage does not carry out-of-sample.** The `sharpe` rule uses the same
   return signal the optimizer maximizes, so the 100% recovery is partly circular. Optimizing on
   trailing mean returns does not generalize, with or without clustering.
5. **Nothing beat holding the whole universe** (Sharpe 1.27) in the rolling test.

## Limits
- The "500-stock" universe is really 368-380 stocks. Tickers with missing data were dropped.
- Ticker lists are hand-built from today's index members, so results carry **survivorship bias**.
- QAOA ran only at S&P 100 scale (20 qubits). Larger runs used the exact best portfolio among the
  representatives, which QAOA matched in the smaller tests.
- Greedy search matched the exact optimum on every S&P 100 case, so **no quantum advantage is claimed**.
- 14 folds is still a modest sample.

## Next steps
- Replace raw trailing mean returns with a more robust estimator (shrinkage or a risk-only objective).
- Use a survivorship-free universe.
- Compare against the Investment Components approach.

## Where to look
- Full write-up: `cluster_qpo/README.md`
- Logs: `cluster_qpo/testing.py/logs/` (key ones: `run_real_data`, `run_validation`, `run_sweep_*`, `run_rwf_sharpe`)
- Plots: `cluster_qpo/testing.py/` (`dendrogram.png`, `heatmap.png`, `cluster_members.png`)
- Result tables: `cluster_qpo/testing.py/*.csv`
