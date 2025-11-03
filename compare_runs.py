# ===============================================
# 🧭 compare_runs.py — Comparatif clair des meilleurs modèles xG
# ===============================================

import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiments = [
    "expected-goals_logreg",
    "expected-goals_rf",
    "expected-goals_lgbm",
    "expected-goals_xgb",
]

all_runs = []

for exp_name in experiments:
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"⚠️ Expérience {exp_name} introuvable, ignorée.")
        continue

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs.empty:
        run = runs.iloc[0]
        # on formate joliment les hyperparamètres
        best_params = {
            k.replace("params.model__", ""): v
            for k, v in run.items()
            if k.startswith("params.model__")
        }
        best_params_str = "; ".join(f"{k}={v}" for k, v in best_params.items())

        all_runs.append({
            "Model": exp_name.replace("expected-goals_", "").upper(),
            "Run ID": run["run_id"],
            "Accuracy": round(run.get("metrics.accuracy", 0), 4),
            "ROC-AUC": round(run.get("metrics.roc_auc", 0), 4),
            "Log Loss": round(run.get("metrics.log_loss", 0), 4),
            "Best Params": best_params_str,
        })

df = pd.DataFrame(all_runs)
df = df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)

print("\n🏆 Tableau comparatif des meilleurs modèles :\n")
print(df.to_string(index=False))

# Sauvegarde propre du CSV
output_path = "artifacts/comparatif_meilleurs_modeles.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Tableau sauvegardé dans {output_path}")
