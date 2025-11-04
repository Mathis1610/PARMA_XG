# ===============================================
# ⚽ generate_xg_predictions.py
# ===============================================
# Generates the "xG_pred" column (Expected Goals probability)
# for each shot using the best MLflow model (LightGBM or XGBoost).
# ===============================================

import mlflow
import pandas as pd
from pathlib import Path
import numpy as np
import logging

# -----------------------------------------------------------
# 1️⃣ General configuration
# -----------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Connect to local MLflow server
DATA_PATH = Path("data/shots.xlsx")               # Input dataset
OUTPUT_PATH = Path("data/shots_with_xg.csv")      # Output file with xG predictions

# Configure simple console logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


# -----------------------------------------------------------
# 2️⃣ Retrieve the best run (based on ROC-AUC)
# -----------------------------------------------------------
def get_best_run(experiment_names):
    """Return the best MLflow run (highest ROC-AUC) among the provided experiments."""
    best_run = None
    best_auc = -np.inf

    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logging.warning(f"Experiment {exp_name} not found, skipping.")
            continue

        # Search for the top run in the experiment (sorted by ROC-AUC)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )
        if not runs.empty and runs.iloc[0]["metrics.roc_auc"] > best_auc:
            best_auc = runs.iloc[0]["metrics.roc_auc"]
            best_run = runs.iloc[0]

    if best_run is None:
        raise ValueError("No valid MLflow runs found in the provided experiments.")
    return best_run


# -----------------------------------------------------------
# 3️⃣ Load the best-performing model
# -----------------------------------------------------------
experiments = ["expected-goals_lgbm", "expected-goals_xgb"]
best_run = get_best_run(experiments)

run_id = best_run["run_id"]
auc = best_run["metrics.roc_auc"]
model_type = best_run["tags.model"]

logging.info(f"🏆 Best model found: {model_type.upper()} (AUC={auc:.4f})")
logging.info(f"🔗 Run ID: {run_id}")

# Load the model directly from MLflow artifacts
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)


# -----------------------------------------------------------
# 4️⃣ Load the dataset and generate predictions
# -----------------------------------------------------------
logging.info(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

# Keep the same column logic as in train.py
target_col = "is_goal"
if target_col in df.columns:
    X = df.drop(columns=[target_col])
else:
    X = df.copy()

# Predict the probability of scoring (Expected Goals)
logging.info("⚙️ Generating xG probabilities...")
df["xG_pred"] = model.predict_proba(X)[:, 1]


# ===============================================
# 🏷️ Automatically create match identifiers
# ===============================================
df = df.reset_index(drop=True)

# Detect the start of new matches:
# whenever the period switches from 2 → 1, start a new match
df["match_id"] = (df["period_id"].shift(1) == 2) & (df["period_id"] == 1)
df["match_id"] = df["match_id"].cumsum() + 1  # increment progressively

print(df[["period_id", "min", "sec", "match_id"]].head(20))


# -----------------------------------------------------------
# 5️⃣ Save the predictions
# -----------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
logging.info(f"✅ File saved: {OUTPUT_PATH.resolve()}")

# Display a small preview
print(df[["x", "y", "xG_pred", "is_goal"]].head())


# ===============================================
# ⚽ Visualization of shots and xG (realistic pitch)
# ===============================================
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

shots = df.copy()

fig, ax = plt.subplots(figsize=(10, 6))
plt.gca().set_facecolor("#E6E6E6")

# Draw a simple football pitch
plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color="black", linewidth=2)

# Goal on the right side (x = 100)
plt.plot([100, 100], [36.8, 63.2], color="red", linewidth=4, label="Opponent Goal")

# Plot the shots: color = goal/no goal, size = xG value
sns.scatterplot(
    data=shots,
    x="x", y="y",
    hue="is_goal",
    size="xG_pred",
    sizes=(20, 400),
    palette={0: "dodgerblue", 1: "red"},
    alpha=0.6,
    ax=ax
)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.title("Shot Map with xG Probabilities", fontsize=14)
plt.xlabel("Position X (0 = own goal, 100 = opponent goal)")
plt.ylabel("Position Y (pitch width)")
plt.legend(title="Goal scored", loc="upper left", frameon=True)
plt.tight_layout()

# Save the visualization
output_img = Path("artifacts") / "visuals"
output_img.mkdir(parents=True, exist_ok=True)
save_path = output_img / "xg_shotmap.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Visualization saved: {save_path.resolve()}")

plt.show()
