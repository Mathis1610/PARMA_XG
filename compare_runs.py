# ===============================================
# 🧭 compare_runs.py — Clear comparison of the best xG models
# ===============================================
# This script retrieves the best run (based on ROC-AUC)
# from several MLflow experiments (LogReg, RF, LGBM, XGB)
# and summarizes their performance metrics in a table.
# ===============================================

import mlflow
import pandas as pd

# Connect to the local MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# List of experiment names to compare
experiments = [
    "expected-goals_logreg",
    "expected-goals_rf",
    "expected-goals_lgbm",
    "expected-goals_xgb",
]

all_runs = []

# Loop through each experiment and extract its best run
for exp_name in experiments:
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"⚠️ Experiment {exp_name} not found, skipping.")
        continue

    # Retrieve the top run based on ROC-AUC score
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs.empty:
        run = runs.iloc[0]

        # Nicely format the model hyperparameters
        best_params = {
            k.replace("params.model__", ""): v
            for k, v in run.items()
            if k.startswith("params.model__")
        }
        best_params_str = "; ".join(f"{k}={v}" for k, v in best_params.items())

        # Store the key metrics for comparison
        all_runs.append({
            "Model": exp_name.replace("expected-goals_", "").upper(),
            "Run ID": run["run_id"],
            "Accuracy": round(run.get("metrics.accuracy", 0), 4),
            "ROC-AUC": round(run.get("metrics.roc_auc", 0), 4),
            "Log Loss": round(run.get("metrics.log_loss", 0), 4),
            "Best Params": best_params_str,
        })

# Build a summary DataFrame sorted by ROC-AUC
df = pd.DataFrame(all_runs)
df = df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

print("\n🏆 Comparison table of the best models:\n")
print(df.to_string(index=False))

# Save results as a clean CSV file
output_path = "artifacts/comparatif_meilleurs_modeles.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Summary saved to {output_path}")
