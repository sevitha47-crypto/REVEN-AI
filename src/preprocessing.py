"""
Data cleaning, missing value handling, outlier detection, and quality scoring.
"""

import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean raw dataframe: handle missing values, duplicates, and negative/invalid metrics.
    Returns cleaned dataframe and cleaning summary report.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Duplicates
    duplicate_rows = int(df_clean.duplicated().sum())
    if duplicate_rows > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        
    # 2. Missing values handling
    missing_before = int(df_clean.isna().sum().sum())
    
    # Numerical imputation with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
            
    # Categorical imputation with mode / 'Unknown'
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col != "Date" and df_clean[col].isna().any():
            mode_series = df_clean[col].mode()
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)
            
    # 3. Validity constraints
    if "Quantity" in df_clean.columns:
        df_clean["Quantity"] = df_clean["Quantity"].clip(lower=1).astype(int)
        
    if "Unit_Price" in df_clean.columns:
        df_clean["Unit_Price"] = df_clean["Unit_Price"].clip(lower=0.01)
        
    if "Discount" in df_clean.columns:
        df_clean["Discount"] = df_clean["Discount"].clip(lower=0.0, upper=0.99)
        
    if "Revenue" in df_clean.columns:
        # Re-verify consistent revenue
        df_clean["Revenue"] = (df_clean["Quantity"] * df_clean["Unit_Price"] * (1.0 - df_clean["Discount"])).round(2)
        
    cleaning_summary = {
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "duplicates_removed": duplicate_rows,
        "missing_imputed": missing_before,
    }
    
    return df_clean, cleaning_summary


def detect_outliers_iqr(df: pd.DataFrame, column: str = "Revenue") -> dict:
    """Detect statistical outliers in a specified numerical column using IQR."""
    if column not in df.columns or df[column].empty:
        return {"outlier_count": 0, "outlier_percentage": 0.0, "lower_bound": 0, "upper_bound": 0}
        
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    count = len(outliers)
    pct = (count / len(df)) * 100.0 if len(df) > 0 else 0.0
    
    return {
        "column": column,
        "outlier_count": count,
        "outlier_percentage": round(pct, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2)
    }


def compute_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate an objective enterprise Data Quality Score (0 - 100%)
    based on completeness, uniqueness, consistency, and validity.
    """
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    completeness_score = 100.0 * (1.0 - (missing_cells / max(1, total_cells)))
    
    duplicate_rows = int(df.duplicated().sum())
    uniqueness_score = 100.0 * (1.0 - (duplicate_rows / max(1, len(df))))
    
    # Validity: check invalid quantities or negative revenues
    invalid_rows = 0
    if "Quantity" in df.columns:
        invalid_rows += int((df["Quantity"] <= 0).sum())
    if "Revenue" in df.columns:
        invalid_rows += int((df["Revenue"] < 0).sum())
    validity_score = 100.0 * (1.0 - (invalid_rows / max(1, len(df))))
    
    # Composite score
    composite_score = round(0.40 * completeness_score + 0.30 * uniqueness_score + 0.30 * validity_score, 1)
    
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    
    return {
        "quality_score": composite_score,
        "completeness_score": round(completeness_score, 1),
        "uniqueness_score": round(uniqueness_score, 1),
        "validity_score": round(validity_score, 1),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "invalid_rows": invalid_rows,
        "numerical_fields": len(num_cols),
        "categorical_fields": len(cat_cols),
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }
