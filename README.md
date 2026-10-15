# Clustering Approach — Portfolio Optimization
## `cluster_qpo` Module

This module implements a **correlation-based hierarchical clustering pipeline** that compresses a large universe of assets (e.g. 500 stocks) down to a small set of representative assets (e.g. 20), which are then fed into a **QUBO → QAOA** quantum portfolio optimizer.

---

## 📌 Table of Contents

1. [How It Works — Big Picture](#1-how-it-works--big-picture)
2. [File-by-File Explanation](#2-file-by-file-explanation)
3. [Full Pipeline Flow](#3-full-pipeline-flow)
4. [How to Run Tests](#4-how-to-run-tests)
5. [How to Run the Real Data Experiment](#5-how-to-run-the-real-data-experiment)
6. [How to Run the Full QAOA Experiment](#6-how-to-run-the-full-qaoa-experiment)
7. [How to Read the Reports](#7-how-to-read-the-reports)
8. [How to Generate Plots](#8-how-to-generate-plots)
9. [Benchmark — Clustering vs Full Universe](#9-benchmark--clustering-vs-full-universe)
10. [Selection Analysis — Why Was This Asset Picked?](#10-selection-analysis--why-was-this-asset-picked)
11. [Environment Setup](#11-environment-setup)
12. [What Each Metric Means](#12-what-each-metric-means)
13. [Validation Against the True Optimum (NEW)](#13-validation-against-the-true-optimum)
14. [Experiments We Ran & Results (NEW)](#14-experiments-we-ran--results)
15. [Key Findings & Honest Caveats (NEW)](#15-key-findings--honest-caveats)
16. [How to Reproduce the Validation Runs (NEW)](#16-how-to-reproduce-the-validation-runs)
17. [Full Timeline of What We Tried](#17-full-timeline-of-what-we-tried)

---

## 1. How It Works — Big Picture

### The Problem
Quantum portfolio optimizers like QAOA have a hard limit on the number of assets they can handle (typically 20–30 qubits). But real portfolios have 100–500+ assets.

### The Solution
**Clustering Compression:**

```
500 Assets (Full Universe)
        ↓
Correlation Matrix
        ↓
Distance Matrix  (sqrt((1 - corr) / 2))
        ↓
Hierarchical Clustering  (Ward/HRP)
        ↓
20 Clusters  (similar stocks grouped together)
        ↓
1 Representative per Cluster  (centroid or best Sharpe)
        ↓
20 Representative Assets
        ↓
μ (Expected Returns)  +  Σ (Covariance Matrix)
        ↓
QUBO Builder
        ↓
QAOA Solver
        ↓
Selected Portfolio (e.g. 5 assets from 20 representatives)
```

### The Key Insight
- **Correlated assets carry redundant information**
- Replacing a cluster of 25 correlated bank stocks with **1 representative bank stock** loses very little portfolio information
- The QAOA then selects the best **combination of cluster representatives** — effectively choosing across all 500 original assets in a compressed way

### What Makes a Good Clustering?
| Metric | Good Value | Meaning |
|---|---|---|
| Intra-cluster correlation | > 0.60 | Stocks inside each cluster are similar |
| Inter-cluster correlation | < 0.40 | Clusters are distinct from each other |
| Representative correlation | > 0.65 | Rep accurately captures its cluster |
| Sharpe loss vs full universe | < 10% | Little information was destroyed |

---

## 2. File-by-File Explanation

### `correlation.py`
**What it does:** Computes the correlation and distance matrices from raw returns.

| Function | Input | Output | Purpose |
|---|---|---|---|
| `covariance_matrix(returns)` | returns DataFrame | Σ matrix | Raw covariance |
| `correlation_matrix(returns)` | returns DataFrame | Corr matrix | Pearson correlation |
| `correlation_to_distance(corr)` | corr matrix | dist matrix | Converts corr → distance for clustering |
| `expected_returns(returns)` | returns DataFrame | μ Series | Annualised mean returns |
| `sharpe_like_scores(returns)` | returns DataFrame | scores Series | Mean/std per asset |

> **Why distance = sqrt((1 - corr) / 2)?**
> Clustering algorithms need a proper distance metric: 0 for identical assets, increasing as assets become less alike, and obeying the triangle inequality (plain `1 - corr` does not). Correlation of +1 (identical) maps to distance 0, correlation 0 maps to about 0.71, and correlation -1 (perfectly opposite) maps to 1. This is the standard HRP (López de Prado) distance.

---

### `hrp.py`
**What it does:** Builds the hierarchical clustering tree and assigns each asset to a cluster.

| Function | Input | Output | Purpose |
|---|---|---|---|
| `hierarchical_clustering(dist, method)` | distance matrix | linkage matrix | Builds the clustering tree |
| `create_clusters(linkage, assets, n)` | linkage matrix | pd.Series (asset→cluster) | Cuts tree into N clusters |
| `cluster_members(labels)` | cluster labels | dict (cluster→assets) | Inverts the mapping |
| `cluster_summary(labels)` | cluster labels | DataFrame | Table of cluster sizes |

> **What is Ward linkage?**
> Ward's method minimises the total within-cluster variance at each merge step. It tends to produce compact, evenly-sized clusters — ideal for portfolio compression.

---

### `representatives.py`
**What it does:** Selects one representative asset per cluster.

| Function | Input | Output | Purpose |
|---|---|---|---|
| `centroid_representative(corr, assets)` | corr matrix, cluster | ticker str | Picks asset closest to cluster average |
| `sharpe_representative(returns, assets)` | returns, cluster | ticker str | Picks asset with highest Sharpe |
| `representatives_by_centroid(corr, clusters)` | corr matrix, clusters | DataFrame | All reps using centroid |
| `representatives_by_sharpe(returns, clusters)` | returns, clusters | DataFrame | All reps using Sharpe |
| `extract_representative_universe(returns, reps)` | returns, reps | reduced returns | Filters returns to reps only |

> **Centroid vs Sharpe:**
> - **Centroid** → picks the most "typical" stock in the cluster. Good for representation accuracy.
> - **Sharpe** → picks the best-performing stock. Good for maximising quality.

---

### `pipeline.py`
**What it does:** Combines all steps into a single function call.

```python
result = run_clustering_reduction(
    returns=returns,
    n_clusters=20,
    method="centroid"    # or "sharpe"
)
# result["clusters"]        → dict of cluster → assets
# result["representatives"] → DataFrame of one rep per cluster
# result["reduced_returns"] → filtered returns DataFrame
```

---

### `reduction.py`
**What it does:** Extends the pipeline to also compute `μ` and `Σ` — exactly what the QUBO builder needs.

```python
result = build_reduced_portfolio_inputs(
    returns=returns,
    n_clusters=20,
    representative_method="centroid"
)
# result["mu"]              → pd.Series shape (20,)
# result["sigma"]           → pd.DataFrame shape (20, 20)
# result["returns"]         → reduced returns
# result["representatives"] → rep table
# result["clusters"]        → cluster mapping
```

---

### `clustered_portfolio.py`
**What it does:** Object-oriented wrapper. Creates a `ClusteredPortfolioProblem` object that holds all state and exposes clean getter methods for the QUBO builder.

```python
problem = ClusteredPortfolioProblem(
    returns=returns_df,
    n_clusters=20,
    representative_method="centroid"
)

problem.get_mu()              # → pd.Series (20,)
problem.get_sigma()           # → pd.DataFrame (20, 20)
problem.get_asset_names()     # → ['AAPL', 'MSFT', ...]
problem.get_representatives() # → DataFrame
problem.get_cluster_mapping() # → dict
problem.summary()             # → dict with compression stats
```

---

### `evaluation.py`
**What it does:** Answers "Did clustering destroy information?" with quantitative metrics.

| Function | Output | Answers |
|---|---|---|
| `intra_cluster_correlation()` | per-cluster corr | Are stocks inside each cluster similar? |
| `inter_cluster_correlation()` | single float | Are clusters distinct from each other? |
| `representative_reconstruction_quality()` | per-cluster score | Does the rep capture its cluster? |
| `variance_retained()` | 0–1 float | How much variance survived compression? |
| `average_cluster_size()` | float | How aggressive was the reduction? |
| `evaluate_clustering()` | full report dict | All of the above in one call |

---

### `backtest.py`
**What it does:** Portfolio performance metrics for full vs reduced universe.

| Function | Output |
|---|---|
| `portfolio_metrics(returns, weights)` | Expected return, volatility, Sharpe, max drawdown, diversification ratio |
| `backtest_representative_universe(returns, n_clusters)` | Metrics on reduced universe |
| `compare_reduced_vs_full(returns, n_clusters)` | Side-by-side comparison table |
| `equal_weight_portfolio(n)` | Equal weight pd.Series |

---

### `benchmark.py`
**What it does:** High-level benchmark report combining universe stats + cluster quality + portfolio metrics.

```python
report = benchmark_clustering_pipeline(
    full_returns=returns,
    reduced_returns=reduced_returns,
    clustering_report=clustering_report
)
print_benchmark_report(report)
```

---

### `selection_analysis.py`
**What it does:** Explains why QAOA selected each asset and how much of the original universe is covered.

| Function | Answers |
|---|---|
| `cluster_coverage()` | How many original assets are represented? |
| `average_cluster_size_selected()` | How big are the selected clusters? |
| `correlation_diversity()` | How uncorrelated are the selected assets? |
| `why_selected(asset)` | Why was this specific stock chosen? |
| `selection_analysis()` | Full report |
| `print_selection_report()` | Human-readable output |

---

### `plots.py`
**What it does:** Visualises the clustering results.

| Function | Output |
|---|---|
| `plot_dendrogram()` | Hierarchical tree of asset relationships |
| `plot_correlation_heatmap()` | Colour-coded correlation matrix sorted by cluster |
| `plot_cluster_members()` | Bar chart: assets by cluster, reps highlighted in red |

---

### `run_clustered_portfolio.py`
**What it does:** End-to-end experiment runner that connects clustering → QUBO → QAOA.

```python
result = run_clustered_portfolio_experiment(
    returns=returns,
    n_clusters=20,
    cardinality=5,
    representative_method="centroid",
    p=1,
    shots=2048
)
```

---

## 3. Full Pipeline Flow

```
returns_df (pd.DataFrame)
│
├── correlation.py
│   ├── correlation_matrix()     → corr
│   └── correlation_to_distance() → dist
│
├── hrp.py
│   ├── hierarchical_clustering(dist) → linkage
│   ├── create_clusters(linkage)      → labels
│   └── cluster_members(labels)       → clusters dict
│
├── representatives.py
│   ├── representatives_by_centroid() → reps DataFrame
│   └── extract_representative_universe() → reduced_returns
│
├── correlation.py
│   ├── expected_returns(reduced) → μ
│   └── covariance_matrix(reduced) → Σ
│
├── clustered_portfolio.py
│   └── ClusteredPortfolioProblem → problem object
│
├── qpo/problem/portfolio.py
│   └── PortfolioProblem(spec) → QUBO
│
└── qpo/solvers/qaoa.py
    └── QAOASolver.run() → selected assets
```

---

## 4. How to Run Tests

Each module has a dedicated test file. Run any of them with:

```bash
# Open a NEW terminal in VS Code first, then:

# Test correlation functions
python3 cluster_qpo/testing.py/test_corr.py

# Test hierarchical clustering
python3 cluster_qpo/testing.py/test_hrp.py

# Test representative selection
python3 cluster_qpo/testing.py/test_representatives.py

# Test reduced returns extraction
python3 cluster_qpo/testing.py/test_reduced_returns.py

# Test full pipeline
python3 cluster_qpo/testing.py/test_pipeline.py

# Test clustering evaluation metrics
python3 cluster_qpo/testing.py/test_evaluation.py

# Test reduction (mu + sigma output)
python3 cluster_qpo/testing.py/test_reduction.py

# Test portfolio backtest metrics
python3 cluster_qpo/testing.py/test_backtest.py

# Test ClusteredPortfolioProblem class
python3 cluster_qpo/testing.py/test_clustered_portfolio.py

# Test benchmark report
python3 cluster_qpo/testing.py/test_benchmark.py

# Test selection analysis
python3 cluster_qpo/testing.py/test_selection_analysis.py

# Test full QAOA experiment runner
python3 cluster_qpo/testing.py/test_run_clustered_portfolio.py
```

> All unit tests use the sample `cluster_qpo/data/returns.csv` (10 random assets, 252 days). Run them from the project root.
> No internet connection required.

---

## 5. How to Run the Real Data Experiment

This downloads real S&P 100 data (cached after first run) and runs the full clustering pipeline.

```bash
python3 cluster_qpo/testing.py/run_real_data_experiment.py
```

**First run:** Downloads 99 tickers from Yahoo Finance (~30 seconds), caches to:
```
cluster_qpo/data/cache/prices_2019-01-01_2024-01-01_99t.parquet
```

**Subsequent runs:** Loads from cache (~2 seconds).

**To save the full report to a text file:**
```bash
python3 cluster_qpo/testing.py/run_real_data_experiment.py > report.txt 2>&1
```

**To change the date range or number of clusters**, edit these lines in `run_real_data_experiment.py`:
```python
returns = load_returns(
    tickers=SP100,
    start="2019-01-01",   # ← change start date
    end="2024-01-01",     # ← change end date
    cache=True
)

N_CLUSTERS = 20           # ← change number of clusters
```

---

## 6. How to Run the Full QAOA Experiment

This runs the complete pipeline: clustering → QUBO → QAOA → selected portfolio.

```bash
python3 cluster_qpo/testing.py/test_run_clustered_portfolio.py
```

**To customise:**
```python
result = run_clustered_portfolio_experiment(
    returns=returns,
    n_clusters=20,       # ← clusters to form
    cardinality=5,       # ← assets to select (K)
    representative_method="centroid",  # or "sharpe"
    p=1,                 # ← QAOA circuit depth
    shots=2048,          # ← shots per evaluation
    optimizer="cobyla",  # ← classical optimizer
    maxiter=200,         # ← max iterations
    verbose=False        # ← True to see optimizer progress
)
```

---

## 7. How to Read the Reports

### Clustering Quality Report
```
Mean Intra-Cluster Correlation    0.6421   ← Target: > 0.60 ✅
Inter-Cluster Correlation         0.4276   ← Target: < 0.50 ✅
Mean Representative Correlation   0.6763   ← Target: > 0.65 ✅
Variance Retained                 0.1832   ← Expected low at 5x compression
Average Cluster Size              4.9500   ← ~5 stocks per cluster
Num Clusters                     20        ← As configured
Original Universe Size           99        ← S&P 100 (minus delisted)
Reduced Universe Size            20        ← 80% compression
```

### Information Loss Report (Full vs Reduced)
```
                       Full Universe  Reduced Universe  Difference  % Change
expected_return             0.211985          0.195535   -0.016450     -7.76
volatility                  0.212558          0.203956   -0.008602     -4.05
sharpe_ratio                0.997300          0.958700   -0.038600     -3.87  ← only 3.87% Sharpe loss!
max_drawdown               -0.344300         -0.318500    0.025800      7.49  ← drawdown improved!
diversification_ratio       1.512500          1.499900   -0.012600     -0.83
```

> **How to interpret:** A Sharpe loss of < 5% for 80% universe compression is excellent. It means the clustering preserved nearly all portfolio quality.

---

## 8. How to Generate Plots

Plots are automatically generated when you run the real data experiment. They are saved as PNG files:

```
cluster_qpo/testing.py/dendrogram.png      ← hierarchical tree
cluster_qpo/testing.py/heatmap.png         ← correlation heatmap
cluster_qpo/testing.py/cluster_members.png ← asset assignments
```

**To generate plots manually in your own script:**

```python
from cluster_qpo.plots import (
    plot_dendrogram,
    plot_correlation_heatmap,
    plot_cluster_members
)

# Dendrogram
plot_dendrogram(
    linkage_matrix=lm,
    assets=list(returns.columns),
    title="My Portfolio Clustering",
    save_path="my_dendrogram.png"   # omit to just display
)

# Correlation Heatmap
plot_correlation_heatmap(
    corr_matrix=corr,
    title="Correlation Matrix",
    cluster_labels=labels,          # sorts by cluster
    save_path="my_heatmap.png"
)

# Cluster Members
plot_cluster_members(
    cluster_labels=labels,
    representatives=reps,           # highlights reps in red
    title="Cluster Assignments",
    save_path="my_clusters.png"
)
```

---

## 9. Benchmark — Clustering vs Full Universe

```python
from cluster_qpo.benchmark import (
    benchmark_clustering_pipeline,
    print_benchmark_report
)

report = benchmark_clustering_pipeline(
    full_returns=returns,
    reduced_returns=reduced_returns,
    clustering_report=clustering_report
)

print_benchmark_report(report)
```

**Output:**
```
CLUSTERING BENCHMARK REPORT
── Universe Statistics ──
  full_assets              : 99
  reduced_assets           : 20
  compression_ratio        : 0.202
  assets_removed           : 79
  reduction_pct            : 79.8

── Portfolio Metrics ──
  [full_universe]
    sharpe_ratio     : 0.9973
  [reduced_universe]
    sharpe_ratio     : 0.9587
```

---

## 10. Selection Analysis — Why Was This Asset Picked?

```python
from cluster_qpo.selection_analysis import (
    selection_analysis,
    print_selection_report
)

analysis = selection_analysis(
    selected_assets=["MSFT", "JPM", "XOM"],
    clusters=problem.clusters,
    representatives=problem.representatives,
    returns=returns
)

print_selection_report(analysis)
```

**Output:**
```
── Cluster Coverage ──
  Assets Represented  : 34 / 99
  Coverage Ratio      : 34.34%

── Asset Explanations ──
  MSFT  (Cluster 7)
    Members    : ['AAPL', 'MSFT', 'NVDA', ...]
    Reason     : Strong centroid representative
                 (mean intra-cluster corr = 0.82)
```

---

## 11. Environment Setup

**Requirements:**
```bash
pip install -r cluster_qpo/requirements.txt
```

**Environment variable** — scripts use `from cluster_qpo...` and `from qpo...`, so the folder that *contains* `cluster_qpo/` and `qpo/` must be on `PYTHONPATH`. Run this from that folder:
```bash
export PYTHONPATH=$(pwd)          # Windows PowerShell: $env:PYTHONPATH = (Get-Location).Path
```

See `SETUP.md` at the project root for the full install steps (venv, requirements, PYTHONPATH) and how to verify the install.

**Verify setup:**
```bash
python3 -c "import pandas, numpy, scipy, yfinance, matplotlib, seaborn; print('All OK ✅')"
python3 cluster_qpo/testing.py/test_corr.py
```

---

## 12. What Each Metric Means

| Metric | Formula | Good Value | Meaning |
|---|---|---|---|
| **Intra-cluster correlation** | Avg pairwise corr within cluster | > 0.60 | Stocks in same cluster are similar |
| **Inter-cluster correlation** | Avg pairwise corr between reps | < 0.40 | Clusters are distinct |
| **Rep reconstruction quality** | Corr(rep, cluster members) | > 0.65 | Rep accurately represents cluster |
| **Variance retained** | Var(reduced) / Var(full) | Depends on compression | Information preserved |
| **Compression ratio** | reduced / full assets | 0.20 for 5x | Reduction achieved |
| **Sharpe ratio** | (return - rf) / volatility | Higher is better | Risk-adjusted return |
| **Max drawdown** | Worst peak-to-trough loss | Less negative is better | Downside risk |
| **Diversification ratio** | Weighted avg vol / portfolio vol | > 1.0 | Benefit of diversification |
| **Coverage ratio** | Assets represented / total | Higher is better | Universe coverage of selection |
| **Diversity score** | 1 - mean pairwise corr | Higher is better | How uncorrelated selection is |

---

## 13. Validation Against the True Optimum

The early checks (Sections 7 and 9) compared an **equal-weight portfolio of all assets** with an **equal-weight portfolio of the representatives**. That only shows the representatives look like the universe. It does **not** show that the optimizer's final K-asset pick is any good.

The real question is:

> If the optimizer must pick exactly K assets, how close does the clustered pipeline get to the best K-asset portfolio in the *full* universe?

### Method (`validation.py`)

Every portfolio is scored on the **same full-universe μ and Σ** with the **same risk aversion q**, using the same objective as the QUBO (exactly K holdings):

```
objective(x) = q * x'Σx  -  μ'x
```

| Compared portfolio | How it is produced |
|---|---|
| **Reference optimum** | Exact brute force over all C(N, K) portfolios when feasible, otherwise greedy + 1-swap local search |
| **Greedy (full)** | Classical greedy heuristic on the full universe |
| **Best of representatives** | Exact best K-subset of the cluster representatives |
| **Cluster + QAOA** | QAOA on the representatives' QUBO |
| **Random K-portfolio** | Average of random K-asset portfolios (sanity floor) |

**% of optimum recovered** = how much of the distance from the random average to the reference optimum a portfolio covers. 100% matches the optimum. 0% is no better than random.

### Functions added to `validation.py`

| Function | Purpose |
|---|---|
| `score_portfolio()` | Objective, annual return, volatility and Sharpe of an equal-weight K-portfolio on full μ/Σ |
| `exact_optimum()` | Exact search over all C(N,K) portfolios (batched, vectorised) |
| `greedy_selection()` | Greedy classical baseline |
| `local_search_refine()` | 1-swap improvement for when exact search is infeasible |
| `reference_optimum()` | Picks exact search or greedy+swap automatically (limit: 30M combinations) |
| `random_baseline()` | Distribution of random K-portfolio objectives |
| `validate_clustering_vs_optimum()` | Full comparison for one K, including a real QAOA run |
| `sweep_representatives()` | Sweeps K x representative method x n_clusters (no QAOA) |
| `walk_forward_validation()` | Rolling train/test out-of-sample test |
| `print_validation_report()` | Readable report |

`run_clustered_portfolio_experiment()` also gained an optional `risk_aversion` argument so QAOA and the reference use the same q.

### Why the exact search is only exact for small K

| Universe | K=3 | K=4 | K=5 |
|---|---|---|---|
| 99 assets | 156,849 (exact) | 3.76M (exact) | 71.5M (exact, a few minutes) |
| ~380 assets | about 9M (exact) | about 8.3 billion (greedy+swap) | about 6.4 trillion (greedy+swap) |

Where greedy+swap is used, the reference is a strong baseline, **not** a proven optimum. On the 99-asset universe greedy matched the exact optimum in every case.

---

## 14. Experiments We Ran & Results

### Experiment A: `centroid` representatives on S&P 100 (with real QAOA)

99 assets reduced to 20 clusters, QAOA with p=2, 4096 shots, 200 iterations.

| K | True optimum | Cluster + QAOA | Sharpe: optimum vs QAOA | Objective recovered | Overlap |
|---|---|---|---|---|---|
| 3 | COST, LLY, NVDA | MSFT, REGN, COST | 1.64 vs 1.22 | 53.6% | 1/3 |
| 4 | COST, LLY, NVDA, WM | GD, MSFT, REGN, COST | 1.63 vs 1.21 | 53.5% | 1/4 |
| 5 | COST, LLY, NOC, NVDA, TSLA | XOM, MSFT, REGN, COST, TMO | 1.67 vs 1.20 | 51.7% | 1/5 |

QAOA found exactly the best portfolio available **among the representatives** in all three cases. The loss came entirely from the clustering step. Centroid selection picks the most *typical* stock per cluster, while the best portfolios need *standout* stocks (NVDA, LLY, TSLA).

### Experiment B: `sharpe` representatives on S&P 100 (with real QAOA)

Same setup, with `representative_method="sharpe"`.

| K | Cluster + QAOA | Objective recovered | Overlap |
|---|---|---|---|
| 3 | NVDA, LLY, COST | 100% | 3/3 |
| 4 | NVDA, LLY, COST, WM | 100% | 4/4 |
| 5 | NOC, NVDA, TSLA, LLY, COST | 100% | 5/5 |

QAOA reproduced the exact true optimum at every K.

### Experiment C: method x n_clusters sweep, S&P 100

Mean % of reference optimum recovered (average over K=3,4,5). Output: `testing.py/sweep_sp100.csv`.

| Method | 20 clusters | 30 clusters | 40 clusters |
|---|---|---|---|
| centroid | 52.9% | 69.6% | 93.6% |
| sharpe | **100%** | **100%** | **100%** |

### Experiment D: method x n_clusters sweep, larger universe (~380 stocks)

Output: `testing.py/sweep_sp500.csv`. References: K=3 exact (LLY, SMCI, TSLA). K=4 and K=5 use greedy+swap (TSLA, SMCI, LLY, PWR[, MCK]).

| Method | 20 clusters | 40 clusters | 60 clusters | 80 clusters |
|---|---|---|---|---|
| centroid | 22.2% | 64.4% | 55.7% | 72.2% |
| sharpe | 90.7% | **100%** | **100%** | **100%** |

With `centroid`, more clusters did **not** improve recovery smoothly (64% at 40 clusters, 56% at 60).

No QAOA was run on this universe. The sweep uses the exact best-of-representatives, which QAOA matched in Experiments A and B. More than about 20 clusters is also beyond a statevector simulator.

### Experiment E: walk-forward (out-of-sample) test, larger universe

`validation.walk_forward_validation()` with 504 training days and 252 test days per fold. Each fold estimates μ/Σ and clusters on the training window only, picks portfolios there, and scores them on the unseen test year. Output: `testing.py/walk_forward_sp500.csv`.

Only **2 folds** fit in the data (test windows 2021-05-26 to 2022-05-24 and 2022-05-25 to 2023-05-25).

Mean out-of-sample Sharpe across the two folds (method = `sharpe`):

| K | Clusters | Cluster portfolio | Reference optimum | Random K-portfolio | Equal-weight universe |
|---|---|---|---|---|---|
| 3 | 20 | 0.27 | 0.22 | 0.19 | 0.23 |
| 3 | 40 | **0.52** | 0.22 | 0.19 | 0.23 |
| 3 | 60 | **0.52** | 0.22 | 0.19 | 0.23 |
| 5 | 20 | **-0.10** | 0.08 | 0.18 | 0.23 |
| 5 | 40 | **0.42** | 0.08 | 0.18 | 0.23 |
| 5 | 60 | **0.42** | 0.08 | 0.18 | 0.23 |

Per fold, the cluster portfolio at K=3, 40 clusters scored 0.220 (fold 1) and 0.811 (fold 2). The reference scored 0.220 and 0.223.

> ⚠️ Only 2 folds. See Experiment F below — this result did not hold up.

### Experiment F: rolling walk-forward, many folds (6-month windows)

Experiment E only had 2 folds, which is not enough to trust. `walk_forward_validation()` gained a `step_days` argument so the test window can roll forward independently of its length. Using 10 years of data (2014–2024), 504 training days, a 126-day (6-month) test window, rolled forward by 126 days, gives **14 non-overlapping test folds** (Oct 2016 → Oct 2023) on 368 stocks. Run for both `sharpe` and `centroid` representatives. Output: `testing.py/rolling_wf_sp500_sharpe.csv` / `rolling_wf_sp500_centroid.csv` (+ `_summary.csv`).

**Mean out-of-sample Sharpe across 14 folds:**

| K | Clusters | Cluster (`sharpe`) | Cluster (`centroid`) | In-sample reference optimum | Random K-portfolio | Equal-weight universe |
|---|---|---|---|---|---|---|
| 3 | 20 | 1.06 | 0.80 | 1.01 | 0.88 | 1.27 |
| 3 | 40 | 1.00 | 1.06 | 1.01 | 0.88 | 1.27 |
| 3 | 60 | 1.00 | 0.79 | 1.01 | 0.88 | 1.27 |
| 5 | 20 | 1.07 | 0.83 | 1.06 | 0.95 | 1.27 |
| 5 | 40 | 1.00 | 1.22 | 1.06 | 0.95 | 1.27 |
| 5 | 60 | 1.05 | 1.09 | 1.06 | 0.95 | 1.27 |

Per-fold Sharpe has a standard deviation of about 1.0–1.5 in every row, so none of these mean gaps are large relative to the noise. Win-rate and paired t-test (cluster `sharpe` vs baselines, across the 14 folds):

| K | Clusters | win vs random | win vs universe | win vs ref | p (vs random) | p (vs universe) |
|---|---|---|---|---|---|---|
| 3 | 20 | 50.0% | 42.9% | 28.6% | 0.55 | 0.52 |
| 3 | 40 | 50.0% | 35.7% | 0.0% | 0.66 | 0.38 |
| 3 | 60 | 50.0% | 35.7% | 0.0% | 0.66 | 0.38 |
| 5 | 20 | 50.0% | 42.9% | 42.9% | 0.68 | 0.47 |
| 5 | 40 | 50.0% | 42.9% | 28.6% | 0.81 | 0.25 |
| 5 | 60 | 50.0% | 50.0% | 21.4% | 0.65 | 0.36 |

All p-values are well above 0.05. The one `centroid` row with p=0.046 (K=3, 60 clusters, vs universe) is a single hit out of about 24 comparisons and is most likely chance.

---

## 15. Key Findings & Honest Caveats

### Findings

1. **The representative rule matters more than the cluster count.** `centroid` loses a lot of the optimum, while `sharpe` recovers essentially all of it in-sample.
2. **QAOA is not the weak link.** In Experiments A and B it matched the exact best portfolio among the representatives every time. Quality is decided by the clustering and representative step.
3. **The earlier equal-weight comparison was too easy a test.** A 3.87% Sharpe loss for 99 to 20 assets (Section 7) hid a roughly 47% loss in optimizer-chosen quality with `centroid`.
4. **The 2-fold out-of-sample result (Experiment E) did not replicate at 14 folds (Experiment F).** With more folds, cluster portfolios won only ~50% of the time against random K-portfolios and ~36–50% against the equal-weight universe — a coin flip, not an edge. All p-values were above 0.25.
5. **`sharpe` vs `centroid` makes no reliable difference out-of-sample**, even though `sharpe` recovers 100% of the optimum in-sample and `centroid` only ~52%. This is the clearest sign that the in-sample recovery numbers are measuring look-ahead, not real skill.
6. **Clustering itself doesn't hurt out-of-sample.** Cluster portfolios scored about the same as the full-universe in-sample reference optimum (1.0–1.2 vs 1.01–1.06). The compression step is not the problem — the problem is that optimizing on trailing mean returns doesn't generalize, with or without clustering.
7. **Nothing beat just holding the whole universe.** The equal-weight universe (1.27 mean Sharpe) was ahead of every clustered/optimized variant in Experiment F.

### Caveats (please read before quoting any number)

- **The 100% in-sample recovery is partly circular.** `sharpe` picks representatives using the same in-sample return signal the optimizer maximizes, so it was bound to find the same stocks. Experiment F confirms this: the in-sample advantage vanishes out-of-sample.
- **The "500" universe is about 368–380 assets** depending on the date range. Tickers get dropped by the loader when they have >10% missing data (delisted, IPO'd mid-window, etc).
- **Survivorship bias.** The ticker list is today's index members, which flatters historical results (including the 1.27 equal-weight universe Sharpe). The list is also hand-built and may not match the official index exactly at each historical date.
- **QAOA ran only at S&P 100 scale** (20 qubits). Larger universes and all walk-forward experiments used the exact best-of-representatives (which QAOA matched exactly in Experiments A/B), not QAOA itself.
- **K=4 and K=5 references on the large universe are heuristic (greedy+swap)**, not proven optima.
- **No quantum advantage claim.** Greedy matched the exact optimum on every S&P 100 case, so classical methods solve these instances easily. The bottleneck is the mean-variance objective's out-of-sample generalization, not the solver.
- **Return estimates are noisy.** Mean daily returns over 1 to 5 years are a weak signal, so any in-sample optimum is partly luck — this is exactly what Experiment F demonstrates empirically.
- **14 folds is still not huge**, and test windows are non-overlapping but drawn from overlapping/adjacent market regimes (e.g. 2020 COVID crash, 2022 rate-hike selloff dominate a few folds each).

### Suggested next steps

- Use a more robust return estimator (shrinkage, Black-Litterman, or a risk-only objective) instead of raw trailing mean returns, since Experiment F suggests mean-return optimization doesn't generalize.
- A survivorship-free universe (point-in-time index membership).
- A comparison against the Investment Components approach — still outstanding.
- Re-run Experiment F with overlapping/more frequent folds (e.g. monthly step) for a larger effective sample size.

---

## 16. How to Reproduce the Validation Runs

Run long jobs in the background with unbuffered output (`-u`), otherwise a redirected log looks empty until the job ends.

```bash
# Experiments A/B: QAOA validation vs true optimum (S&P 100)
METHOD=centroid python3 -u cluster_qpo/testing.py/run_validation.py 3 4 5
METHOD=sharpe   python3 -u cluster_qpo/testing.py/run_validation.py 3 4 5

# Experiments C/D: method x n_clusters sweep
python3 -u cluster_qpo/testing.py/run_sweep.py sp100
python3 -u cluster_qpo/testing.py/run_sweep.py sp500

# Experiment E: walk-forward out-of-sample test (2 folds, 2019-2024)
python3 -u cluster_qpo/testing.py/run_walk_forward.py sp500

# Experiment F: rolling walk-forward, many folds (14 folds, 2014-2024, 6-month windows)
python3 -u cluster_qpo/testing.py/run_rolling_walk_forward.py sp500 sharpe
python3 -u cluster_qpo/testing.py/run_rolling_walk_forward.py sp500 centroid
```

To run one in the background and watch the log:

```bash
nohup python3 -u cluster_qpo/testing.py/run_validation.py 3 4 5 > /tmp/val.log 2>&1 &
tail -f /tmp/val.log
```

| Run | Approximate time |
|---|---|
| Validation K=3,4 (S&P 100) | 1 to 2 minutes |
| Validation K=5 (71.5M exact combos) | a few extra minutes |
| Sweep, S&P 100 | under 1 minute |
| Sweep and walk-forward, larger universe | several minutes |
| Rolling walk-forward, 14 folds (Experiment F) | 2 to 3 minutes per method |

The first run downloads prices and caches them as parquet (the 2014-2024 range used by Experiment F is a separate, larger cache file from the 2019-2024 range used elsewhere). Later runs use the cache.

---

## 17. Full Timeline of What We Tried

1. Built the clustering pipeline: correlation, hierarchical (Ward) clustering, representative selection (centroid and Sharpe), reduction to μ/Σ.
2. Added evaluation, backtest, benchmark and selection-analysis layers, plus a `ClusteredPortfolioProblem` wrapper.
3. Connected to the existing QUBO builder and QAOA solver (PennyLane simulator) with `run_clustered_portfolio.py`.
4. Pulled real S&P 100 data with a cached Yahoo Finance loader (`cluster_qpo/data/`).
5. Ran the first real-data experiment: 99 to 20 assets, intra-cluster correlation 0.64, inter-cluster 0.43, about 3.9% Sharpe loss on an equal-weight comparison. Clusters were sensible (energy, banks, tech and others grouped together).
6. Realised that comparison did not involve the optimizer, and built a true-optimum validation with exact brute force (Experiment A).
7. Found `centroid` recovers only about 52% of the optimum, because it picks typical rather than standout stocks.
8. Switched to `sharpe` representatives and swept cluster counts (Experiments B, C).
9. Scaled to a larger universe with a scalable reference solver (exact at K=3, greedy+swap at K=4,5) (Experiment D).
10. Added a walk-forward train/test split to check out-of-sample behaviour (Experiment E) — but only 2 folds fit, so the result (cluster beats reference/random) was not trustworthy on its own.
11. Added a `step_days` parameter to decouple the roll step from the test window length, pulled 2014-2024 data, and re-ran as a 14-fold rolling walk-forward for both `sharpe` and `centroid` (Experiment F). This is what overturned Experiment E: out-of-sample, cluster portfolios won only ~50% of folds against random, and `sharpe` vs `centroid` showed no reliable difference despite the huge in-sample gap.

### Problems hit along the way (and fixes)

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: cluster_qpo` | Set `PYTHONPATH` to the folder containing `cluster_qpo/` (and `qpo/`) |
| `ModuleNotFoundError: pandas` / other packages | Activate the venv and run `pip install -r cluster_qpo/requirements.txt` |
| Backend class name mismatch | Use `PennyLaneSimulator` |
| Result key mismatch (`reduced_returns` vs `returns`) | Fixed in `backtest.py` |
| `MMC` ticker had no price data | Removed from the S&P 100 list |
| Parquet caching failed | Installed `pyarrow` |
| Background logs appeared empty | Run with `python3 -u` |

---

## 📁 Complete File Structure

```
cluster_qpo/
├── __init__.py
├── correlation.py          ← Step 1: Build corr & distance matrices
├── hrp.py                  ← Step 2: Hierarchical clustering
├── representatives.py      ← Step 3: Select one rep per cluster
├── pipeline.py             ← Combines steps 1–3 in one call
├── reduction.py            ← Adds μ, Σ for QUBO
├── clustered_portfolio.py  ← OOP wrapper for full problem
├── evaluation.py           ← Cluster quality metrics
├── backtest.py             ← Portfolio performance metrics
├── benchmark.py            ← Full benchmark report
├── selection_analysis.py   ← Why was each asset selected?
├── plots.py                ← Dendrogram, heatmap, cluster chart
├── run_clustered_portfolio.py ← End-to-end QAOA experiment
├── validation.py           ← True-optimum, sweep and walk-forward validation (NEW)
└── testing.py/ (NEW)
    ├── test_corr.py
    ├── test_hrp.py
    ├── test_representatives.py
    ├── test_reduced_returns.py
    ├── test_pipeline.py
    ├── test_evaluation.py
    ├── test_reduction.py
    ├── test_backtest.py
    ├── test_clustered_portfolio.py
    ├── test_benchmark.py
    ├── test_selection_analysis.py
    ├── test_run_clustered_portfolio.py
    ├── run_real_data_experiment.py  ← Main real-data experiment
    ├── run_validation.py            ← QAOA vs true optimum (METHOD env var) (NEW)
    ├── run_sweep.py                 ← method x n_clusters sweep (NEW)
    ├── run_walk_forward.py          ← out-of-sample walk-forward, 2 folds (NEW)
    ├── run_rolling_walk_forward.py  ← rolling walk-forward, many folds (NEW)
    ├── sweep_sp100.csv              ← results (NEW)
    ├── sweep_sp500.csv              ← results (NEW)
    ├── walk_forward_sp500.csv       ← results, Experiment E (NEW)
    ├── rolling_wf_sp500_sharpe.csv        ← results, Experiment F (NEW)
    ├── rolling_wf_sp500_sharpe_summary.csv   ← results, Experiment F (NEW)
    ├── rolling_wf_sp500_centroid.csv      ← results, Experiment F (NEW)
    └── rolling_wf_sp500_centroid_summary.csv ← results, Experiment F (NEW)
```

```
cluster_qpo/data/
├── __init__.py
├── universe.py             ← SP100, SP500, NIFTY50 ticker lists
├── yfinance_loader.py      ← Download, cache, compute returns
└── cache/
    └── prices_*.parquet    ← Cached price data (auto-generated)
```
