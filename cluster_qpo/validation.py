# qpo/cluster_qpo/validation.py
#
# Validation: is "cluster -> QAOA" as good as the TRUE optimum?
#
#   TRUE OPTIMUM : exact brute force over ALL C(N, K) portfolios
#                  of the FULL universe (e.g. 99 assets).
#   QAOA PICK    : K assets chosen by QAOA from the ~20 cluster
#                  representatives.
#
# Both portfolios are scored on the SAME full-universe mu / Sigma and the
# SAME risk aversion q, so the comparison is apples to apples:
#
#       objective(x) = q * x' Sigma x  -  mu' x       (exactly K holdings)

from itertools import combinations, islice
from math import comb

import numpy as np
import pandas as pd

from .correlation import expected_returns, covariance_matrix


# ---------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------
def score_portfolio(
    tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float,
    periods_per_year: int = 252
) -> dict:
    """Score an equal-weight K-asset portfolio on the full-universe mu/Sigma."""
    k = len(tickers)
    x = pd.Series(0.0, index=mu.index)
    x[tickers] = 1.0

    mu_v = mu.values
    sig_v = sigma.values
    xv = x.values

    objective = float(risk_aversion * xv @ sig_v @ xv - mu_v @ xv)

    w = xv / k
    ret = float(mu_v @ w) * periods_per_year
    vol = float(np.sqrt(w @ sig_v @ w)) * np.sqrt(periods_per_year)

    return {
        "tickers": list(tickers),
        "objective": objective,
        "annual_return": ret,
        "annual_volatility": vol,
        "sharpe": ret / vol if vol > 0 else 0.0,
    }


# ---------------------------------------------------------------
# Exact brute force (vectorised, batched)
# ---------------------------------------------------------------
def exact_optimum(
    mu: pd.Series,
    sigma: pd.DataFrame,
    budget: int,
    risk_aversion: float,
    batch_size: int = 1_000_000,
    max_combinations: int = 100_000_000,
    verbose: bool = True
) -> dict:
    """
    Exact global optimum over all C(N, K) portfolios.

    Uses index gathering instead of building 0/1 matrices, so
    C(99, 4) = 3.7M combos takes seconds and C(99, 5) = 71.5M
    takes a few minutes.
    """
    n = len(mu)
    total = comb(n, budget)
    if total > max_combinations:
        raise ValueError(f"C({n},{budget}) = {total:,} exceeds max_combinations")

    mu_v = mu.values
    sig_v = sigma.values
    q = risk_aversion

    best_obj = np.inf
    best_combo = None
    done = 0

    it = combinations(range(n), budget)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        idx = np.array(chunk, dtype=np.int32)

        lin = mu_v[idx].sum(axis=1)
        quad = np.zeros(len(idx))
        for a in range(budget):
            for b in range(budget):
                quad += sig_v[idx[:, a], idx[:, b]]

        obj = q * quad - lin
        j = int(np.argmin(obj))
        if obj[j] < best_obj:
            best_obj = float(obj[j])
            best_combo = idx[j].copy()

        done += len(idx)
        if verbose and total > batch_size:
            print(f"    [brute force K={budget}] {done:,}/{total:,}", end="\r")

    if verbose and total > batch_size:
        print()

    tickers = [mu.index[i] for i in best_combo]
    return {
        "tickers": tickers,
        "objective": best_obj,
        "n_evaluated": total,
    }


def greedy_selection(
    mu: pd.Series,
    sigma: pd.DataFrame,
    budget: int,
    risk_aversion: float
) -> list[str]:
    """Classical greedy heuristic on the full universe (reference)."""
    n = len(mu)
    mu_v, sig_v = mu.values, sigma.values
    chosen: list[int] = []
    for _ in range(budget):
        best_i, best_e = None, np.inf
        for i in range(n):
            if i in chosen:
                continue
            x = np.zeros(n)
            x[chosen + [i]] = 1.0
            e = risk_aversion * x @ sig_v @ x - mu_v @ x
            if e < best_e:
                best_i, best_e = i, e
        chosen.append(best_i)
    return [mu.index[i] for i in chosen]


def local_search_refine(
    mu: pd.Series,
    sigma: pd.DataFrame,
    start_tickers: list[str],
    risk_aversion: float,
    max_rounds: int = 50
) -> list[str]:
    """1-swap local search: swap one held asset for one outside asset
    while the objective improves. Strong reference when exact is infeasible."""
    n = len(mu)
    mu_v, sig_v = mu.values, sigma.values
    pos = {t: i for i, t in enumerate(mu.index)}
    cur = [pos[t] for t in start_tickers]

    def obj(idx):
        x = np.zeros(n)
        x[idx] = 1.0
        return risk_aversion * x @ sig_v @ x - mu_v @ x

    best = obj(cur)
    for _ in range(max_rounds):
        improved = False
        for slot in range(len(cur)):
            for cand in range(n):
                if cand in cur:
                    continue
                trial = cur.copy()
                trial[slot] = cand
                e = obj(trial)
                if e < best - 1e-15:
                    best, cur, improved = e, trial, True
        if not improved:
            break
    return [mu.index[i] for i in cur]


def reference_optimum(
    mu: pd.Series,
    sigma: pd.DataFrame,
    budget: int,
    risk_aversion: float,
    max_exact: int = 30_000_000,
    verbose: bool = True
) -> dict:
    """Exact optimum when C(N,K) <= max_exact, otherwise greedy + swap
    local search (a strong but NOT proven-optimal reference)."""
    if comb(len(mu), budget) <= max_exact:
        res = exact_optimum(
            mu, sigma, budget, risk_aversion,
            max_combinations=max_exact, verbose=verbose
        )
        res["exact"] = True
        return res

    g = greedy_selection(mu, sigma, budget, risk_aversion)
    tickers = local_search_refine(mu, sigma, g, risk_aversion)
    sc = score_portfolio(tickers, mu, sigma, risk_aversion)
    return {
        "tickers": tickers,
        "objective": sc["objective"],
        "n_evaluated": None,
        "exact": False,
    }


def sweep_representatives(
    returns: pd.DataFrame,
    budgets=(3, 4, 5),
    methods=("centroid", "sharpe"),
    cluster_counts=(20, 30, 40),
    verbose: bool = True
) -> pd.DataFrame:
    """
    How much of the reference optimum survives clustering, for each
    (K, representative method, n_clusters)?

    Uses the exact best-of-representatives (no QAOA): QAOA matched this
    exactly in the earlier validation, and >20 clusters is beyond a
    statevector simulator anyway.
    """
    from qpo.problem.portfolio import balanced_risk_aversion
    from .pipeline import run_clustering_reduction

    mu = expected_returns(returns)
    sigma = covariance_matrix(returns)
    rows = []

    for K in budgets:
        q = float(balanced_risk_aversion(mu.values, sigma.values, K))
        ref = reference_optimum(mu, sigma, K, q, verbose=False)
        ref_score = score_portfolio(ref["tickers"], mu, sigma, q)
        rnd = random_baseline(mu, sigma, K, q)
        denom = rnd["mean"] - ref_score["objective"]
        if verbose:
            kind = "exact" if ref["exact"] else "greedy+swap"
            print(f"[sweep] K={K} reference ({kind}): {ref['tickers']}")

        for method in methods:
            for nc in cluster_counts:
                res = run_clustering_reduction(
                    returns, n_clusters=nc, method=method
                )
                reps = list(res["reduced_returns"].columns)
                best = exact_optimum(
                    mu[reps], sigma.loc[reps, reps], K, q, verbose=False
                )
                sc = score_portfolio(best["tickers"], mu, sigma, q)
                rows.append({
                    "K": K,
                    "method": method,
                    "n_clusters": nc,
                    "% of reference recovered": round(
                        (rnd["mean"] - sc["objective"]) / denom * 100, 1
                    ),
                    "sharpe": round(sc["sharpe"], 3),
                    "ref_sharpe": round(ref_score["sharpe"], 3),
                    "overlap": len(set(best["tickers"]) & set(ref["tickers"])),
                    "assets": ", ".join(best["tickers"]),
                })
    return pd.DataFrame(rows)


def _oos_metrics(test_returns: pd.DataFrame, tickers: list[str]) -> dict:
    """Equal-weight out-of-sample metrics on the test window."""
    r = test_returns[tickers].mean(axis=1)
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    return {
        "oos_return": ann_ret,
        "oos_vol": ann_vol,
        "oos_sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
    }


def walk_forward_validation(
    returns: pd.DataFrame,
    budgets=(3, 5),
    cluster_counts=(20, 40, 60),
    method: str = "sharpe",
    train_days: int = 504,
    test_days: int = 252,
    step_days: int = None,
    n_random: int = 300,
    seed: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Rolling walk-forward test. For each fold:
        1. estimate mu/Sigma on the TRAIN window only
        2. pick portfolios on train:
             - reference optimum (exact if feasible, else greedy+swap)
             - cluster (train-only) -> exact best of representatives
        3. score every pick on the unseen TEST window
    Baselines on the same test window: equal-weight whole universe and the
    average random K-asset portfolio.
    """
    from qpo.problem.portfolio import balanced_risk_aversion
    from .pipeline import run_clustering_reduction

    rng = np.random.default_rng(seed)
    n_days = len(returns)
    rows = []
    start = 0
    fold = 0
    while start + train_days + test_days <= n_days:
        train = returns.iloc[start:start + train_days]
        test = returns.iloc[start + train_days:start + train_days + test_days]
        fold += 1
        label = f"{test.index[0].date()}..{test.index[-1].date()}"
        mu = expected_returns(train)
        sigma = covariance_matrix(train)

        univ = _oos_metrics(test, list(returns.columns))

        # cluster once per (n_clusters) on train data only
        reps_by_nc = {}
        for nc in cluster_counts:
            res = run_clustering_reduction(train, n_clusters=nc, method=method)
            reps_by_nc[nc] = list(res["reduced_returns"].columns)

        for K in budgets:
            q = float(balanced_risk_aversion(mu.values, sigma.values, K))
            ref = reference_optimum(mu, sigma, K, q, verbose=False)
            ref_m = _oos_metrics(test, ref["tickers"])

            rand_sharpes = []
            cols = list(returns.columns)
            for _ in range(n_random):
                pick = list(rng.choice(cols, size=K, replace=False))
                rand_sharpes.append(_oos_metrics(test, pick)["oos_sharpe"])

            base = {
                "fold": fold, "test_window": label, "K": K,
                "universe_ew_sharpe": round(univ["oos_sharpe"], 3),
                "random_mean_sharpe": round(float(np.mean(rand_sharpes)), 3),
                "ref_oos_sharpe": round(ref_m["oos_sharpe"], 3),
                "ref_assets": ", ".join(ref["tickers"]),
            }
            for nc, reps in reps_by_nc.items():
                best = exact_optimum(
                    mu[reps], sigma.loc[reps, reps], K, q, verbose=False
                )
                m = _oos_metrics(test, best["tickers"])
                rows.append({
                    **base,
                    "n_clusters": nc,
                    "cluster_oos_sharpe": round(m["oos_sharpe"], 3),
                    "cluster_oos_return": round(m["oos_return"], 3),
                    "cluster_assets": ", ".join(best["tickers"]),
                })
            if verbose:
                print(f"[walk-forward] fold {fold} ({label}) K={K} done", flush=True)
        start += step_days or test_days
    return pd.DataFrame(rows)


def random_baseline(
    mu: pd.Series,
    sigma: pd.DataFrame,
    budget: int,
    risk_aversion: float,
    n_samples: int = 5000,
    seed: int = 0
) -> dict:
    """Distribution of objectives for random K-asset portfolios."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    mu_v, sig_v = mu.values, sigma.values
    objs = np.empty(n_samples)
    for s in range(n_samples):
        x = np.zeros(n)
        x[rng.choice(n, size=budget, replace=False)] = 1.0
        objs[s] = risk_aversion * x @ sig_v @ x - mu_v @ x
    return {"mean": float(objs.mean()), "std": float(objs.std()), "all": objs}


# ---------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------
def validate_clustering_vs_optimum(
    returns: pd.DataFrame,
    budget: int = 3,
    n_clusters: int = 20,
    representative_method: str = "centroid",
    risk_aversion="auto",
    qaoa_kwargs: dict = None,
    verbose: bool = True
) -> dict:
    """
    Compare, for a given K:
        1. TRUE optimum (exact brute force on the full universe)
        2. Greedy heuristic on the full universe
        3. QAOA selection from the clustered representative universe
        4. Average random K-portfolio (sanity floor)

    All scored on full-universe mu / Sigma with the same q.
    """
    from qpo.problem.portfolio import balanced_risk_aversion
    from .run_clustered_portfolio import run_clustered_portfolio_experiment

    mu = expected_returns(returns)
    sigma = covariance_matrix(returns)

    # One shared q so objectives are comparable
    if risk_aversion == "auto":
        q = float(balanced_risk_aversion(mu.values, sigma.values, budget))
    else:
        q = float(risk_aversion)

    # 1. True optimum
    if verbose:
        print(f"[validation] Exact search: C({len(mu)},{budget}) = "
              f"{comb(len(mu), budget):,} portfolios")
    opt = reference_optimum(mu, sigma, budget, q, verbose=verbose)
    opt_score = score_portfolio(opt["tickers"], mu, sigma, q)

    # 2. Greedy
    greedy_tickers = greedy_selection(mu, sigma, budget, q)
    greedy_score = score_portfolio(greedy_tickers, mu, sigma, q)

    # 3. Cluster + QAOA
    kwargs = dict(qaoa_kwargs or {})
    exp = run_clustered_portfolio_experiment(
        returns=returns,
        n_clusters=n_clusters,
        cardinality=budget,
        representative_method=representative_method,
        risk_aversion=q,
        **kwargs
    )
    qaoa_tickers = list(exp["selected_assets"])
    qaoa_feasible = len(qaoa_tickers) == budget
    qaoa_score = (
        score_portfolio(qaoa_tickers, mu, sigma, q)
        if qaoa_tickers else None
    )

    # 4. Best achievable from the representatives (exact, reduced universe)
    reps = list(exp["problem"].get_asset_names())
    mu_r, sig_r = mu[reps], sigma.loc[reps, reps]
    cluster_opt = exact_optimum(mu_r, sig_r, budget, q, verbose=False)
    cluster_opt_score = score_portfolio(cluster_opt["tickers"], mu, sigma, q)

    # 5. Random floor
    rnd = random_baseline(mu, sigma, budget, q)

    # Ranking of each result among random portfolios
    def percentile(obj):
        return float((rnd["all"] > obj).mean() * 100)

    def gap(obj):
        # How much of the distance (random mean -> optimum) is recovered
        denom = rnd["mean"] - opt_score["objective"]
        return float((rnd["mean"] - obj) / denom * 100) if denom != 0 else 0.0

    rows = [
        ("True optimum (full, exact)", opt_score),
        ("Greedy (full)", greedy_score),
        ("Best of representatives (exact)", cluster_opt_score),
        ("Cluster + QAOA", qaoa_score),
    ]
    table = pd.DataFrame([
        {
            "method": name,
            "assets": ", ".join(s["tickers"]),
            "objective": round(s["objective"], 6),
            "annual_return": round(s["annual_return"], 4),
            "annual_vol": round(s["annual_volatility"], 4),
            "sharpe": round(s["sharpe"], 3),
            "% of optimum recovered": round(gap(s["objective"]), 1),
            "beats % of random": round(percentile(s["objective"]), 1),
        }
        for name, s in rows if s is not None
    ]).set_index("method")

    overlap = len(set(qaoa_tickers) & set(opt["tickers"]))

    return {
        "budget": budget,
        "risk_aversion": q,
        "optimum": opt_score,
        "greedy": greedy_score,
        "cluster_exact": cluster_opt_score,
        "qaoa": qaoa_score,
        "qaoa_feasible": qaoa_feasible,
        "overlap_with_optimum": overlap,
        "random_mean_objective": rnd["mean"],
        "table": table,
    }


def print_validation_report(result: dict) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 70)
    print(f"VALIDATION  K = {result['budget']}   q = {result['risk_aversion']:.4g}")
    print("=" * 70)
    print(result["table"].to_string())
    print()
    print(f"QAOA feasible (exactly K assets) : {result['qaoa_feasible']}")
    print(f"QAOA assets also in true optimum : "
          f"{result['overlap_with_optimum']} / {result['budget']}")
    print("\n'% of optimum recovered': 100 = matches the true optimum,")
    print("0 = no better than a random K-asset portfolio.\n")
