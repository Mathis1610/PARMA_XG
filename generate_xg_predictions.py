# ===============================================
# ⚽ generate_xg_predictions.py
# ===============================================
# Génère la colonne xG_pred pour chaque tir en utilisant
# le meilleur modèle MLflow (LightGBM ou XGBoost)
# ===============================================

import mlflow
import pandas as pd
from pathlib import Path
import numpy as np
import logging

# -----------------------------------------------------------
# 1️⃣ Configuration générale
# -----------------------------------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")
DATA_PATH = Path("data/shots.xlsx")
OUTPUT_PATH = Path("data/shots_with_xg.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


# -----------------------------------------------------------
# 2️⃣ Charger le meilleur run (selon ROC-AUC)
# -----------------------------------------------------------
def get_best_run(experiment_names):
    """Retourne le meilleur run (AUC max) parmi les expériences données."""
    best_run = None
    best_auc = -np.inf

    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logging.warning(f"Expérience {exp_name} non trouvée, ignorée.")
            continue

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )
        if not runs.empty and runs.iloc[0]["metrics.roc_auc"] > best_auc:
            best_auc = runs.iloc[0]["metrics.roc_auc"]
            best_run = runs.iloc[0]

    if best_run is None:
        raise ValueError("Aucun run valide trouvé dans les expériences MLflow.")
    return best_run


# -----------------------------------------------------------
# 3️⃣ Charger le modèle le plus performant
# -----------------------------------------------------------
experiments = ["expected-goals_lgbm", "expected-goals_xgb"]
best_run = get_best_run(experiments)

run_id = best_run["run_id"]
auc = best_run["metrics.roc_auc"]
model_type = best_run["tags.model"]

logging.info(f"🏆 Meilleur modèle trouvé : {model_type.upper()} (AUC={auc:.4f})")
logging.info(f"🔗 Run ID : {run_id}")

model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# -----------------------------------------------------------
# 4️⃣ Charger les données et faire les prédictions
# -----------------------------------------------------------
logging.info(f"📂 Chargement du dataset : {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

# On garde la même logique de colonnes que dans train.py
target_col = "is_goal"
if target_col in df.columns:
    X = df.drop(columns=[target_col])
else:
    X = df.copy()

# Prédictions de probabilité (Expected Goals)
logging.info("⚙️ Génération des probabilités xG...")
df["xG_pred"] = model.predict_proba(X)[:, 1]

# ===============================================
# 🏷️ Création d'un identifiant de match automatique
# ===============================================
df = df.reset_index(drop=True)
# Détection des débuts de nouveaux matchs :
# chaque fois que la période repasse de 2 → 1
df["match_id"] = (df["period_id"].shift(1) == 2) & (df["period_id"] == 1)
df["match_id"] = df["match_id"].cumsum() + 1  # incrémentation progressive

print(df[["period_id", "min", "sec", "match_id"]].head(20))

# -----------------------------------------------------------
# 5️⃣ Sauvegarde
# -----------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
logging.info(f"✅ Fichier sauvegardé : {OUTPUT_PATH.resolve()}")

# Aperçu
print(df[["x", "y", "xG_pred", "is_goal"]].head())

# ===============================================
# ⚽ Visualisation des tirs et des xG (terrain réaliste)
# ===============================================
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

shots = df.copy()

fig, ax = plt.subplots(figsize=(10, 6))
plt.gca().set_facecolor("#E6E6E6")

# Tracé du terrain simplifié
plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color="black", linewidth=2)

# But à droite (x=100)
plt.plot([100, 100], [36.8, 63.2], color="red", linewidth=4, label="But adverse")

# Représentation des tirs
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

plt.title("Carte des tirs et probabilités xG", fontsize=14)
plt.xlabel("Position X (0 = notre but, 100 = but adverse)")
plt.ylabel("Position Y (largeur du terrain)")
plt.legend(title="But marqué", loc="upper left", frameon=True)
plt.tight_layout()

# Sauvegarde
output_img = Path("artifacts") / "visuals"
output_img.mkdir(parents=True, exist_ok=True)
save_path = output_img / "xg_shotmap.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Graphique sauvegardé : {save_path.resolve()}")

plt.show()

