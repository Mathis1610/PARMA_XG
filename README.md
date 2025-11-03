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
