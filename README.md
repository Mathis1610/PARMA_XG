# ⚽ PARMA_XG – Expected Goals Model

This project builds an **Expected Goals (xG)** model for football shots using **Python** and **MLflow**.  
The objective is to estimate, for each shot, the probability that it becomes a goal.

---

## 🚀 Project Overview

The project includes:
- **`train.py`** – trains and logs machine-learning models (LogReg, Random Forest, LightGBM, XGBoost) with MLflow  
- **`generate_xg_predictions.py`** – loads the best model from MLflow and computes `xG_pred` for each shot  
- **`compare_runs.py`** – compares different MLflow runs (AUC, accuracy, etc.)
- **`shots.xlsx`** (in `/data/`) – dataset containing ~50 000 shots with contextual features  

All experiments and metrics are tracked locally with **MLflow UI** (`http://127.0.0.1:5000`).

---

## 🧠 Model Workflow

1. **Data Loading**  
   The script reads `data/shots.xlsx` and cleans features.

2. **Train/Test Split**  
   80 % of data for training, 20 % for testing.

3. **Model Training**  
   The user selects the model type with `--model`  
   and optionally enables hyperparameter tuning (`--tune`).

4. **Evaluation & Logging**  
   - Metrics: Accuracy, ROC-AUC, Log Loss  
   - Confusion matrix, feature importances, and predictions are saved in `/artifacts/`

5. **xG Prediction Generation**  
   The best MLflow run is reloaded and applied to the full dataset  
   → adds a new column `xG_pred`.

---

## ⚙️ Installation

Clone the repository and create a virtual environment:
```bash
git clone https://github.com/Mathis1610/PARMA_XG.git
cd PARMA_XG

# create & activate venv (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## 🧪 Reproducible Experiments

All experiments are tracked with **MLflow** and can be reproduced exactly using the following commands.  
Each command fixes the random seed and test split to ensure identical results across runs.

### ⚡ LightGBM (with Hyperparameter Tuning)
```bash
python train.py --data-path data/shots.xlsx \
                --model lgbm \
                --run-name "LGBM tuned" \
                --tune \
                --test-size 0.2 \
                --random-state 42
```

### 🌳 XGBoost (with Hyperparameter Tuning)
```bash
python train.py --data-path data/shots.xlsx \
                --model xgb \
                --run-name "XGB tuned" \
                --tune \
                --test-size 0.2 \
                --random-state 42
```

Both commands:
- Train the selected model on 80% of the dataset and evaluate it on 20%
- Log all metrics, parameters, and artifacts (confusion matrix, feature importances, etc.) to MLflow
- Store outputs inside the `artifacts/` folder

---

## 📊 Compare the Best Models

After training multiple models, you can easily compare their best runs:
```bash
python compare_runs.py
```

This script extracts, for each model type:
- The best run ID (highest ROC-AUC)
- The main metrics (Accuracy, ROC-AUC, Log Loss)
- The best hyperparameters

→ It automatically generates a CSV file: `artifacts/comparatif_meilleurs_modeles.csv`

---

## ⚽ Generate xG Predictions

Once the best model (LightGBM or XGBoost) is identified, you can apply it to the full dataset:
```bash
python generate_xg_predictions.py
```

This script:
- Loads the top MLflow model (highest AUC)
- Computes the probability of scoring (`xG_pred`) for each shot
- Saves a new file: `data/shots_with_xg.csv`
- Generates a visual shot map (`artifacts/visuals/xg_shotmap.png`)

---

## 📈 Explore MLflow Dashboard

To visualize all experiments, metrics, and artifacts, launch the MLflow UI:
```bash
mlflow ui
```

Then open your browser at `http://127.0.0.1:5000`

---