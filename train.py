"""Train an expected goals (xG) classifier and log the run with MLflow.

This script expects an Excel or CSV dataset containing shot level data and a
binary target column that indicates if the shot resulted in a goal. Features
are inferred automatically based on their data types so the script can be used
with datasets that share the same schema but different values.
"""
# .\.venv\Scripts\activate

# =====================================================
# Configuration MLflow
# =====================================================

from __future__ import annotations
import os


import mlflow
print("🎯 Tracking URI actuel :", mlflow.get_tracking_uri())

os.makedirs("mlruns", exist_ok=True)

# =====================================================
# Imports
# =====================================================
import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.utils import shuffle
from time import time
from tqdm import tqdm
import mlflow
import logging

@dataclass
class DatasetSplit:
    """Container for dataset splits."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


DEFAULT_POSITIVE_STRINGS = {"goal", "goals", "scored", "yes", "true", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/shots.xlsx"),
        help=(
            "Path to the input dataset. Both Excel (.xlsx, .xls) and CSV files are "
            "supported."
        ),
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="is_goal",
        help="Name of the binary target column that indicates if the shot resulted in a goal.",
    )
    parser.add_argument(
        "--positive-class",
        type=str,
        default=None,
        help=(
            "Optional value that represents the positive class. If omitted, the script "
            "will attempt to infer it from the data."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use as the hold-out test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for data splitting and model training.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the logistic regression solver.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Defaults to a local ./mlruns directory if omitted.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="expected-goals",
        help="Name of the MLflow experiment where runs will be recorded.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name for easier identification.",
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        nargs="*",
        default=None,
        help="List of column names that should be removed before training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "rf", "lgbm", "xgb"],
        help="Choice of model: 'logreg', 'rf', 'lgbm', or 'xgb'.",
    )
    parser.add_argument(
    "--tune",
    action="store_true",
    help="If set, performs hyperparameter search for the selected model."
    )
    
    return parser.parse_args()



def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not locate dataset at {path}. Provide the correct path using --data-path."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        logging.info("Loading dataset from Excel file %s", path)
        return pd.read_excel(path)
    if suffix == ".csv":
        logging.info("Loading dataset from CSV file %s", path)
        return pd.read_csv(path)

    raise ValueError(
        "Unsupported file format. Provide an Excel (.xlsx/.xls) or CSV file."
    )


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: Optional[Iterable[str]] = None,
    positive_class: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    df_clean = df.copy()

    if drop_columns:
        missing = [col for col in drop_columns if col not in df_clean.columns]
        if missing:
            raise KeyError(
                f"Columns to drop not found in dataset: {missing}. Available columns: {list(df_clean.columns)}"
            )
        df_clean = df_clean.drop(columns=list(drop_columns))

    target_series = df_clean.pop(target_column)

    if target_series.isna().any():
        logging.info("Dropping %d rows with missing target values", target_series.isna().sum())
        mask = target_series.notna()
        target_series = target_series[mask]
        df_clean = df_clean.loc[mask]

    target_series = coerce_target_to_binary(target_series, positive_class)

    feature_df = df_clean

    all_na_rows = feature_df.isna().all(axis=1)
    if all_na_rows.any():
        logging.info("Dropping %d rows with all-null features", all_na_rows.sum())
        feature_df = feature_df.loc[~all_na_rows]
        target_series = target_series.loc[feature_df.index]

    bool_cols = df_clean.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        logging.info("Converting boolean columns to int: %s", list(bool_cols))
        df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    int_cols = df_clean.select_dtypes(include=["int", "int64", "int32"]).columns
    if len(int_cols) > 0:
        logging.info("Converting integer columns to float to avoid MLflow warning.")
        df_clean[int_cols] = df_clean[int_cols].astype(float)

    return feature_df, target_series


def coerce_target_to_binary(series: pd.Series, positive_class: Optional[str]) -> pd.Series:
    if series.nunique() != 2:
        raise ValueError(
            "Target column must contain exactly two unique values after cleaning."
        )

    if positive_class is not None:
        normalized_positive = str(positive_class).lower()
        unique_values = series.astype(str).str.lower().unique()
        if normalized_positive not in unique_values:
            raise ValueError(
                f"Positive class '{positive_class}' not found in target column. Available values: {unique_values}"
            )
        mapping = {
            value: 1 if value == normalized_positive else 0
            for value in unique_values
        }
        logging.info("Mapping target values using explicit positive class '%s'", positive_class)
        return series.astype(str).str.lower().map(mapping)

    if series.dtype.kind in {"i", "u", "b", "f"}:
        unique = sorted(series.dropna().unique())
        if len(unique) != 2:
            raise ValueError("Numeric target column must contain exactly two unique values.")
        mapping = {unique[0]: 0, unique[1]: 1}
        logging.info("Mapping numeric target values %s to %s", unique, mapping)
        return series.map(mapping)

    series_lower = series.astype(str).str.lower()
    unique = sorted(series_lower.dropna().unique())

    mapping = {}
    for value in unique:
        if value in DEFAULT_POSITIVE_STRINGS:
            mapping[value] = 1
        elif value in {"0", "no", "false", "f"}:
            mapping[value] = 0

    if len(mapping) != 2:
        if len(unique) != 2:
            raise ValueError(
                "Unable to infer positive and negative classes automatically."
            )
        logging.warning(
            "Inferring class mapping based on alphabetical order for values %s", unique
        )
        mapping = {unique[0]: 0, unique[1]: 1}

    logging.info("Mapping categorical target values %s to %s", unique, mapping)
    return series_lower.map(mapping)


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model_name: str,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    #  Sélection du modèle
    if model_name == "logreg":
        model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=random_state)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "lgbm":
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "xgb":
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in df.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns

def tune_model(pipeline, model_name, x_train, y_train, random_state):
    """
    Advanced hyperparameter tuning with progress tracking and MLflow logging.

    This function abstracts the tuning strategy for each supported model family:
      - Logistic Regression / Random Forest → exhaustive GridSearchCV
      - LightGBM / XGBoost → randomized search over a wider space (RandomizedSearchCV)

    Why two strategies?
    - Grid search is fine for small, well-behaved grids (LogReg, RF).
    - Randomized search explores larger spaces more efficiently (LGBM, XGB).

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        End-to-end pipeline (preprocessing + estimator) to tune.
    model_name : str
        One of {"logreg", "rf", "lgbm", "xgb"} to select the search space.
    x_train, y_train : pd.DataFrame, pd.Series
        Training split used for cross-validated search.
    random_state : int
        Seed for reproducibility (used by randomized searches).

    Returns
    -------
    sklearn.Pipeline
        The fitted best estimator (pipeline with best hyperparameters).
    """

    logging.info(f"🎯 Hyperparameter tuning for {model_name.upper()} started...")

    # --- Define model-specific search spaces ---------------------------------
    if model_name == "logreg":
        # Small, interpretable grid for a convex problem → GridSearchCV is fine
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "saga"],
        }
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )

    elif model_name == "rf":
        # RF has a few influential levers; grid is still manageable
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 30],
            "model__max_features": ["sqrt", "log2"],
        }
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )

    elif model_name == "lgbm":
        # Larger space → randomized search explores faster under a fixed budget
        param_dist = {
            "model__num_leaves": [20, 31, 40, 60, 80, 120],
            "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "model__n_estimators": [300, 500, 800, 1200, 1500],
            "model__min_child_samples": [5, 10, 20, 30, 50],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0, 0.1, 0.5, 1],
            "model__reg_alpha": [0, 0.1, 0.5, 1],
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=50,            # budget: increase for deeper search (time ↑)
            cv=10,                # robust CV for imbalanced data + AUC metric
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )

    elif model_name == "xgb":
        # Similar randomized strategy for XGBoost
        param_dist = {
            "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0, 0.1, 0.2, 0.3],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__n_estimators": [300, 500, 800, 1200],
            "model__reg_lambda": [0.5, 1, 2],
            "model__reg_alpha": [0, 0.1, 0.5, 1],
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=50,
            cv=10,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )

    else:
        raise ValueError(f"Tuning not supported for model: {model_name}")

    # --- Run the search and time it ------------------------------------------
    logging.info("🚀 Starting hyperparameter search...")
    start_time = time()
    search.fit(x_train, y_train)
    elapsed = (time() - start_time) / 60

    logging.info(f"✅ Tuning finished in {elapsed:.2f} min")
    logging.info(f"🏆 Best AUC = {search.best_score_:.4f}")
    logging.info(f"🎯 Best params = {search.best_params_}")

    # --- Track results in MLflow ---------------------------------------------
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("cv_best_auc", search.best_score_)
    mlflow.log_metric("tuning_time_min", round(elapsed, 2))

    # Save full CV results (useful for later analysis/repro)
    results_df = pd.DataFrame(search.cv_results_)
    results_path = Path("artifacts") / "tuning_results"
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / f"tuning_{model_name}.csv"
    results_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))
    logging.info(f"📁 Saved all tuning results to {csv_path}")

    # Return the best fitted pipeline (already refit on full training data)
    return search.best_estimator_


def evaluate_model(
    pipeline: Pipeline,
    split: DatasetSplit,
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate the trained pipeline on the test split.

    We compute:
      - ROC-AUC on probabilities (threshold-independent ranking quality)
      - Log Loss (calibration of probabilities)
      - Accuracy (threshold-dependent; less informative for imbalance)
      - Full classification report (precision/recall/F1 per class)

    Returns a (metrics_dict, report_df) pair to both log and inspect.
    """
    # Predict well-calibrated probabilities first (needed for AUC/logloss)
    y_pred_proba = pipeline.predict_proba(split.x_test)[:, 1]

    # Turn probabilities into hard labels at 0.5 (simple baseline threshold)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(split.y_test, y_pred),
        "log_loss": log_loss(split.y_test, y_pred_proba),
        "roc_auc": roc_auc_score(split.y_test, y_pred_proba),
    }

    # Rich per-class report (includes weighted/macro averages)
    report = classification_report(
        split.y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    return metrics, report_df


def log_confusion_matrix(pipeline: Pipeline, split: DatasetSplit, output_dir: Path) -> Path:
    """
    Compute and save a confusion matrix image for the test set.

    Useful to visualize the trade-off between false positives and false negatives
    at the current 0.5 threshold (you may later tune this threshold by maximizing
    F1, Youden’s J, cost-sensitive metrics, etc.).
    """
    # Build the plot from the estimator's predictions
    ConfusionMatrixDisplay.from_estimator(
        pipeline,
        split.x_test,
        split.y_test,
        display_labels=["No Goal", "Goal"],
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Persist to artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved confusion matrix to %s", output_path)
    return output_path


def log_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    output_dir: Path,
) -> Path:
    """
    Export feature importance (tree models) or coefficients (linear models).

    - Logistic Regression exposes `coef_` (weights after preprocessing).
    - Tree-based models (RF/LGBM/XGB) expose `feature_importances_`.
    The function resolves the transformed feature names from the ColumnTransformer
    so the CSV is human-readable and aligned with the one-hot encoded design.
    """

    model = pipeline.named_steps["model"]

    # --- Linear models: coefficients -----------------------------------------
    if hasattr(model, "coef_"):
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                transformed_feature_names = preprocessor.get_feature_names_out(feature_names)
            except TypeError:
                # Some sklearn versions require no args
                transformed_feature_names = preprocessor.get_feature_names_out()
        else:
            transformed_feature_names = feature_names

        coef = model.coef_.ravel()
        importance_df = pd.DataFrame(
            {"feature": transformed_feature_names, "importance": coef}
        )

    # --- Tree models: impurity-based importance -------------------------------
    elif hasattr(model, "feature_importances_"):
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                transformed_feature_names = preprocessor.get_feature_names_out(feature_names)
            except TypeError:
                transformed_feature_names = preprocessor.get_feature_names_out()
        else:
            transformed_feature_names = feature_names

        importance_df = pd.DataFrame(
            {"feature": transformed_feature_names, "importance": model.feature_importances_}
        )

    else:
        raise AttributeError(
            "Model does not expose feature importance or coefficients for analysis."
        )

    # Sort by absolute contribution to surface the most influential features
    importance_df = importance_df.sort_values(
        by="importance", key=lambda s: s.abs(), ascending=False
    )

    # Save to artifacts for easy inspection in MLflow
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    logging.info("Saved feature importance to %s", output_path)
    return output_path


def log_predictions(
    pipeline: Pipeline,
    split: DatasetSplit,
    output_dir: Path,
) -> Path:
    """
    Save per-example predictions for the test split.

    This CSV is extremely useful for:
      - Manual spot-checks (what did we miss?)
      - Threshold tuning / decision analysis
      - Downstream dashboards or error analysis notebooks
    """
    probabilities = pipeline.predict_proba(split.x_test)[:, 1]

    # Include original features to ease later analysis/joins
    predictions_df = split.x_test.copy()
    predictions_df["y_true"] = split.y_test.values
    predictions_df["y_pred"] = (probabilities >= 0.5).astype(int)
    predictions_df["y_proba"] = probabilities

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    logging.info("Saved test predictions to %s", output_path)
    return output_path


def log_artifacts(
    pipeline: Pipeline,
    split: DatasetSplit,
    metrics: dict,
    report_df: pd.DataFrame,
    artifact_dir: Path,
    requirements_path: Optional[Path] = None,
) -> None:
    """
    Log all useful runtime artifacts to MLflow for full reproducibility:

      - metrics.json: compact set of key metrics
      - classification_report.csv: precision/recall/F1 per class
      - confusion_matrix.png: threshold-based error distribution
      - feature_importance.csv: model explainability snapshot
      - test_predictions.csv: row-level outputs for error analysis
      - requirements.txt (optional): environment capture
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1) Key metrics as JSON (easy to diff across runs)
    metrics_path = artifact_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    mlflow.log_artifact(str(metrics_path))

    # 2) Detailed classification report
    report_path = artifact_dir / "classification_report.csv"
    report_df.to_csv(report_path, index=True)
    mlflow.log_artifact(str(report_path))

    # 3) Confusion matrix image
    conf_matrix_path = log_confusion_matrix(pipeline, split, artifact_dir)
    mlflow.log_artifact(str(conf_matrix_path))

    # 4) Feature importance / coefficients
    feature_names = split.x_train.columns.tolist()
    importance_path = log_feature_importance(pipeline, feature_names, artifact_dir)
    mlflow.log_artifact(str(importance_path))

    # 5) Row-level predictions on the test set
    predictions_path = log_predictions(pipeline, split, artifact_dir)
    mlflow.log_artifact(str(predictions_path))

    # 6) Environment capture (optional but recommended)
    if requirements_path and requirements_path.exists():
        mlflow.log_artifact(str(requirements_path), artifact_path="environment")


def main() -> None:
    """
    Orchestrate the full training run:
      1) Parse CLI args and configure logging
      2) Connect to MLflow and set the experiment
      3) Load & preprocess data (X, y)
      4) Build the preprocessing + model pipeline
      5) (Optional) Hyperparameter tuning
      6) Evaluate on the hold-out test set
      7) Log metrics, artifacts, and the model to MLflow
    """
    args = parse_args()

    # If no explicit run name, create a readable timestamped one
    if args.run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{args.model}_run_{timestamp}"

    configure_logging()

    # Close any active MLflow run (defensive)
    mlflow.end_run()

    # Ensure we talk to the expected tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logging.info("✅ Connected to MLflow server at http://127.0.0.1:5000")

    # Create a dedicated experiment per model family for clean comparisons
    experiment_name = f"{args.experiment_name}_{args.model}"
    mlflow.set_experiment(experiment_name)
    logging.info(f"📁 Using MLflow experiment: {experiment_name}")

    # Reproducibility: fix seeds for numpy/random (and model seeds where possible)
    set_reproducibility(args.random_state)

    # --- Load & preprocess ----------------------------------------------------
    df = load_dataset(args.data_path)
    features, target = preprocess_dataset(
        df,
        args.target_column,
        drop_columns=args.drop_columns,
        positive_class=args.positive_class,
    )

    # 1) Infer feature types (numeric vs categorical)
    numeric_features, categorical_features = get_feature_types(features)

    # 2) Build the preprocessing + estimator pipeline for the chosen model
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        args.model,
        args.max_iter,
        random_state=args.random_state,
    )
    mlflow.log_param("model_type", args.model)

    # 3) Train/test split (hold-out for honest evaluation)
    dataset_split = split_dataset(
        features, target, test_size=args.test_size, random_state=args.random_state
    )

    # --- MLflow run scope -----------------------------------------------------
    try:
        # nested=True prevents conflicts if called from a parent run
        with mlflow.start_run(run_name=args.run_name, nested=True):
            # Tags help filter/search runs later
            mlflow.set_tag("model", args.model)
            mlflow.set_tag("dataset", str(args.data_path))

            logging.info(
                "Starting MLflow run with %d training samples and %d test samples",
                len(dataset_split.x_train),
                len(dataset_split.x_test),
            )

            # Log high-level run parameters (data shape, test size, etc.)
            mlflow.log_params(
                {
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "max_iter": args.max_iter,
                    "numeric_feature_count": len(numeric_features),
                    "categorical_feature_count": len(categorical_features),
                }
            )

            # --- Train (with or without tuning) -------------------------------
            if args.tune:
                logging.info("🎯 Hyperparameter tuning activated.")
                pipeline = tune_model(
                    pipeline,
                    args.model,
                    dataset_split.x_train,
                    dataset_split.y_train,
                    args.random_state,
                )
            else:
                logging.info("🚀 Training model without tuning.")
                pipeline.fit(dataset_split.x_train, dataset_split.y_train)

            # Persist final model hyperparameters (after tuning/fit)
            for param_name, value in pipeline.named_steps["model"].get_params().items():
                mlflow.log_param(f"model__{param_name}", value)

            # --- Evaluate & log ----------------------------------------------
            metrics, report_df = evaluate_model(pipeline, dataset_split)
            mlflow.log_metrics(metrics)

            artifacts_dir = Path("artifacts") / mlflow.active_run().info.run_id
            requirements_path = Path("requirements.txt")
            log_artifacts(
                pipeline,
                dataset_split,
                metrics,
                report_df,
                artifact_dir=artifacts_dir,
                requirements_path=requirements_path,
            )

            # --- Register the trained pipeline as an MLflow model ------------
            signature = infer_signature(
                dataset_split.x_test, pipeline.predict_proba(dataset_split.x_test)
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=dataset_split.x_test.head(5),
            )

            logging.info("Run complete. Metrics: %s", metrics)

    finally:
        # Always close the run even if an exception occurs
        mlflow.end_run()


if __name__ == "__main__":
    main()
