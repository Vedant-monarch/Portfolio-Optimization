# qpo/clustering/run_clustered_portfolio.py

import pandas as pd

from .clustered_portfolio import ClusteredPortfolioProblem
from .evaluation import evaluate_clustering
from qpo.problem.portfolio import PortfolioSpec, build_qubo
from qpo.solvers.qaoa import QAOASolver
from qpo.backends.pennylane_sim import PennyLaneSimulator


def run_clustered_portfolio_experiment(
    returns: pd.DataFrame,
    n_clusters: int = 20,
    cardinality: int = 5,
    representative_method: str = "centroid",
    p: int = 1,
    shots: int = 2048,
    optimizer: str = "cobyla",
    maxiter: int = 200,
    verbose: bool = False,
    risk_aversion="auto"
) -> dict:
    """
    End-to-end clustered portfolio experiment.

    Pipeline
    --------
    Full Universe
        ↓
    Clustering (HRP/Ward)
        ↓
    Representative Selection
        ↓
    μ, Σ  →  QUBO  →  QAOA
        ↓
    Selected Representatives

    Parameters
    ----------
    returns : pd.DataFrame
        Full universe returns (rows=dates, cols=assets)
    n_clusters : int
        Number of clusters to form
    cardinality : int
        Portfolio budget K (number of assets to select)
    representative_method : str
        "centroid" or "sharpe"
    p : int
        QAOA circuit depth
    shots : int
        Shots per circuit evaluation
    optimizer : str
        Classical optimizer (cobyla, spsa, etc.)
    maxiter : int
        Max optimizer iterations
    verbose : bool
        Print optimizer progress

    Returns
    -------
    dict with keys:
        - "problem"           : ClusteredPortfolioProblem
        - "clustering_report" : dict
        - "qubo"              : QUBO
        - "solver_result"     : SolverResult
        - "selected_assets"   : list[str]
        - "selected_clusters" : list[int]
    """

    # ---------------------------------------------------------------
    # Step 1 — Build clustered portfolio problem
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — BUILDING CLUSTERED PORTFOLIO PROBLEM")
    print("=" * 60)

    problem = ClusteredPortfolioProblem(
        returns=returns,
        n_clusters=n_clusters,
        representative_method=representative_method
    )
    print(problem)
    print()
    for k, v in problem.summary().items():
        print(f"  {k}: {v}")
    print()

    # ---------------------------------------------------------------
    # Step 2 — Evaluate cluster quality
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 — EVALUATING CLUSTER QUALITY")
    print("=" * 60)

    clustering_report = evaluate_clustering(
        returns=returns,
        clusters=problem.clusters,
        representatives=problem.representatives,
        reduced_returns=problem.reduced_returns
    )

    print(clustering_report["summary"])
    print()

    # ---------------------------------------------------------------
    # Step 3 — Build QUBO
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 3 — BUILDING QUBO")
    print("=" * 60)

    mu    = problem.get_mu().values
    sigma = problem.get_sigma().values
    tickers = problem.get_asset_names()

    spec = PortfolioSpec(
        mu=mu,
        sigma=sigma,
        budget=cardinality,
        risk_aversion=risk_aversion,
        tickers=tickers
    )

    from qpo.problem.portfolio import PortfolioProblem
    portfolio_problem = PortfolioProblem(spec)
    qubo = portfolio_problem.qubo

    print(f"  QUBO size : {qubo.n} × {qubo.n}")
    print(f"  Budget    : {cardinality} assets")
    print()

    # ---------------------------------------------------------------
    # Step 4 — Run QAOA
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 4 — RUNNING QAOA")
    print("=" * 60)

    backend = PennyLaneSimulator()

    solver = QAOASolver(
        problem=portfolio_problem,
        backend=backend,
        p=p,
        shots=shots,
        optimizer=optimizer,
        maxiter=maxiter,
        verbose=verbose
    )

    solver_result = solver.run()
    print(solver_result.report())
    print()

    # ---------------------------------------------------------------
    # Step 5 — Decode selected representatives
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 5 — SELECTED REPRESENTATIVES")
    print("=" * 60)

    solution    = solver_result.solution
    selected_assets = solution.selected

    rep_index = (
        problem.representatives
        .reset_index()
        .set_index("representative")
    )

    selected_clusters = [
        int(rep_index.loc[asset, "cluster_id"])
        if asset in rep_index.index else -1
        for asset in selected_assets
    ]

    for asset, cluster_id in zip(selected_assets, selected_clusters):
        print(f"  {asset}  (Cluster {cluster_id})")

    print()
    print(f"  feasible : {solution.feasible}")
    print(f"  return   : {solution.expected_return:.6f}")
    print(f"  volatility: {solution.volatility:.6f}")
    print(f"  sharpe   : {solution.sharpe_like:.4f}")

    return {
        "problem":           problem,
        "clustering_report": clustering_report,
        "qubo":              qubo,
        "solver_result":     solver_result,
        "selected_assets":   selected_assets,
        "selected_clusters": selected_clusters
    }
