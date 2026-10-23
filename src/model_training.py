"""
Machine learning pipeline: model training, chronological evaluation, model comparison,
and serialization using Joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

from src.feature_engineering import build_ml_features

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sales_model.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.pkl")

CLASSIFICATION_MODEL_FILE = os.path.join(MODEL_DIR, "classification_model.pkl")
CLASSIFICATION_METADATA_FILE = os.path.join(MODEL_DIR, "classification_metadata.pkl")


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str = "Quantity",
    test_ratio: float = 0.20
) -> dict:
    """
    Train 4 regression models using a chronological train/test split.
    Calculates MAE, RMSE, R² on unseen future data.
    Saves the champion model to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Engineer features
    df_featured, feature_cols, cat_cols = build_ml_features(df, target_col=target_col)
    
    # Sort chronologically
    df_featured = df_featured.sort_values("Date").reset_index(drop=True)
    
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    # Chronological Split (Past -> Future)
    split_idx = int(len(df_featured) * (1.0 - test_ratio))
    train_df = df_featured.iloc[:split_idx]
    test_df = df_featured.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )
    
    # Model Candidate Definitions
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_pipelines = {}
    test_predictions = {}
    
    for name, model in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Fit on training partition
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        # Predict on chronological holdout
        preds = pipe.predict(X_test)
        # Ensure non-negative predictions for volume
        preds = np.clip(preds, 1.0, None)
        test_predictions[name] = preds
        
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        
        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        })
        
    results_df = pd.DataFrame(results)
    
    # Best model selection: lowest RMSE (and tie-break highest R²)
    best_row = results_df.sort_values(by=["RMSE", "R2"], ascending=[True, False]).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    
    # Mark Champion
    results_df["Status"] = np.where(results_df["Model"] == best_model_name, "★ Champion", "Evaluated")
    
    # Actual vs Predicted comparison dataframe for testing set
    eval_df = test_df[["Date", "Product", "Category", "Region", target_col]].copy()
    eval_df["Actual"] = y_test.values
    eval_df["Predicted"] = np.round(test_predictions[best_model_name], 1)
    eval_df["Absolute_Error"] = np.round(np.abs(eval_df["Actual"] - eval_df["Predicted"]), 1)
    
    # Save Model & Metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics": results_df.to_dict(orient="records"),
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_col": target_col,
        "split_date": test_df["Date"].min().strftime("%Y-%m-%d")
    }
    
    joblib.dump(best_pipeline, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    
    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "eval_df": eval_df,
        "metadata": metadata,
        "trained_pipelines": trained_pipelines
    }


def load_champion_model() -> tuple:
    """Load serialized champion model and its metadata if available."""
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        try:
            pipeline = joblib.load(MODEL_FILE)
            metadata = joblib.load(METADATA_FILE)
            return pipeline, metadata
        except Exception:
            return None, None
    return None, None


def train_and_evaluate_classification_models(
    df: pd.DataFrame,
    target_col: str = "Demand_Tier",
    test_ratio: float = 0.20
) -> dict:
    """
    Train and evaluate 4 supervised classification models (Random Forest, Gradient Boosting,
    Logistic Regression, and Decision Tree).
    Supports arbitrary classification targets (e.g. Demand Velocity, Churn, Customer Segment, Risk).
    Selects champion model by F1-Score & Accuracy and serializes pipeline + metadata to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df_work = df.copy()

    # If target_col not in dataframe or needs creation
    if target_col not in df_work.columns or target_col == "Demand_Tier":
        # Create balanced 3-tier demand velocity classification
        q33 = df_work["Quantity"].quantile(0.33)
        q66 = df_work["Quantity"].quantile(0.66)
        df_work["Demand_Tier"] = np.where(
            df_work["Quantity"] > q66, "High Velocity",
            np.where(df_work["Quantity"] > q33, "Moderate Velocity", "Low Velocity")
        )
        target_col = "Demand_Tier"
    elif pd.api.types.is_numeric_dtype(df_work[target_col]) and df_work[target_col].nunique() > 10:
        # User selected a continuous column for classification: bin into 3 balanced categories
        q33 = df_work[target_col].quantile(0.33)
        q66 = df_work[target_col].quantile(0.66)
        tier_col = f"{target_col}_Tier"
        df_work[tier_col] = np.where(
            df_work[target_col] > q66, "High",
            np.where(df_work[target_col] > q33, "Medium", "Low")
        )
        target_col = tier_col

    # Clean target: drop NaN or empty strings
    df_work[target_col] = df_work[target_col].astype(str).str.strip()
    df_work = df_work[df_work[target_col].notna() & (df_work[target_col] != "")]
    
    # Feature engineering
    df_featured, feature_cols, cat_cols = build_ml_features(df_work, target_col=target_col)
    
    # Sort chronologically if Date available
    if "Date" in df_featured.columns:
        df_featured = df_featured.sort_values("Date").reset_index(drop=True)
        
    num_cols = [c for c in feature_cols if c not in cat_cols and c != target_col]
    clean_cat_cols = [c for c in cat_cols if c != target_col]

    # Chronological Split (Past -> Future)
    split_idx = int(len(df_featured) * (1.0 - test_ratio))
    train_df = df_featured.iloc[:split_idx]
    test_df = df_featured.iloc[split_idx:]
    
    # Guard against unseen classes in train
    if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 1:
        # Fallback to stratified shuffle if chronological slice lacks class diversity
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_featured, test_size=test_ratio, random_state=42, stratify=df_featured[target_col])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    classes_list = sorted(list(y_train.unique()))

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), clean_cat_cols)
        ],
        remainder="drop"
    )

    candidate_classifiers = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=4, random_state=42
        )
    }

    metrics_list = []
    trained_pipelines = {}
    test_predictions = {}
    test_probabilities = {}

    for model_name, clf in candidate_classifiers.items():
        try:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            
            # Predict probabilities if supported
            probs = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            metrics_list.append({
                "Model": model_name,
                "Accuracy": round(float(acc) * 100, 2),
                "F1_Score": round(float(f1) * 100, 2),
                "Precision": round(float(prec) * 100, 2),
                "Recall": round(float(rec) * 100, 2)
            })

            trained_pipelines[model_name] = pipe
            test_predictions[model_name] = preds
            test_probabilities[model_name] = probs
        except Exception as e:
            continue

    if not metrics_list:
        raise ValueError("Classification model training encountered an issue fitting classes.")

    results_df = pd.DataFrame(metrics_list).sort_values("F1_Score", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    results_df["Status"] = np.where(results_df["Model"] == best_model_name, "★ Champion", "Evaluated")

    # Confusion matrix and detailed report
    best_preds = test_predictions[best_model_name]
    cm = confusion_matrix(y_test, best_preds, labels=classes_list)
    cm_df = pd.DataFrame(cm, index=classes_list, columns=classes_list)

    clf_report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)

    # Evaluation comparison sample
    eval_cols = [c for c in ["Date", "Product", "Category", "Region"] if c in test_df.columns]
    eval_df = test_df[eval_cols].copy()
    eval_df["Actual_Class"] = y_test.values
    eval_df["Predicted_Class"] = best_preds
    eval_df["Match"] = np.where(eval_df["Actual_Class"] == eval_df["Predicted_Class"], "Correct", "Mismatch")

    # Save to disk
    metadata = {
        "best_model_name": best_model_name,
        "target_col": target_col,
        "classes": classes_list,
        "feature_cols": feature_cols,
        "cat_cols": clean_cat_cols,
        "num_cols": num_cols,
        "metrics": results_df.to_dict(orient="records"),
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "classification_report": clf_report
    }

    joblib.dump(best_pipeline, CLASSIFICATION_MODEL_FILE)
    joblib.dump(metadata, CLASSIFICATION_METADATA_FILE)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "eval_df": eval_df,
        "cm_df": cm_df,
        "classes": classes_list,
        "metadata": metadata,
        "trained_pipelines": trained_pipelines
    }


def load_champion_classification_model() -> tuple:
    """Load serialized champion classification model and its metadata if available."""
    if os.path.exists(CLASSIFICATION_MODEL_FILE) and os.path.exists(CLASSIFICATION_METADATA_FILE):
        try:
            pipeline = joblib.load(CLASSIFICATION_MODEL_FILE)
            metadata = joblib.load(CLASSIFICATION_METADATA_FILE)
            return pipeline, metadata
        except Exception:
            return None, None
    return None, None

