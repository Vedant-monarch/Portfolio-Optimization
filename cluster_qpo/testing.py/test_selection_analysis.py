import pandas as pd
from cluster_qpo.run_clustered_portfolio import (
    run_clustered_portfolio_experiment
)
from cluster_qpo.selection_analysis import (
    selection_analysis,
    print_selection_report
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
    maxiter=100,
    verbose=False
)

analysis = selection_analysis(
    selected_assets=result["selected_assets"],
    clusters=result["problem"].clusters,
    representatives=result["problem"].representatives,
    returns=returns
)

print_selection_report(analysis)

print("=== SUMMARY TABLE ===")
print(analysis["summary"])
