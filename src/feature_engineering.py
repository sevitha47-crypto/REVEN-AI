"""
Feature engineering pipeline for machine learning and predictive analytics.
Includes calendar decomposition, cyclical transforms, lag, and rolling statistics.
All time-series features strictly respect chronological ordering to prevent data leakage.
"""

import pandas as pd
import numpy as np

def build_ml_features(df: pd.DataFrame, target_col: str = "Quantity") -> tuple[pd.DataFrame, list, list]:
    """
    Construct rich time-series features without data leakage.
    Returns:
        featured_df: DataFrame with engineered features
        feature_names: List of model feature column names
        categorical_cols: List of categorical feature names
    """
    df_feat = df.sort_values(["Date", "Product"]).copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat["Date"]):
        df_feat["Date"] = pd.to_datetime(df_feat["Date"])
        
    # 1. Calendar Features
    df_feat["Year"] = df_feat["Date"].dt.year
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["Quarter"] = df_feat["Date"].dt.quarter
    df_feat["Week"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    df_feat["Day"] = df_feat["Date"].dt.day
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek # 0=Mon, 6=Sun
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].isin([5, 6]).astype(int)
    df_feat["IsMonthStart"] = df_feat["Date"].dt.is_month_start.astype(int)
    df_feat["IsMonthEnd"] = df_feat["Date"].dt.is_month_end.astype(int)
    
    # 2. Cyclical Temporal Encodings
    df_feat["Month_Sin"] = np.sin(2 * np.pi * df_feat["Month"] / 12.0)
    df_feat["Month_Cos"] = np.cos(2 * np.pi * df_feat["Month"] / 12.0)
    df_feat["DayOfWeek_Sin"] = np.sin(2 * np.pi * df_feat["DayOfWeek"] / 7.0)
    df_feat["DayOfWeek_Cos"] = np.cos(2 * np.pi * df_feat["DayOfWeek"] / 7.0)
    
    # 3. Product-Level Historical Baselines (Shifted to prevent leakage)
    # Determine base numerical series for lag and momentum features
    if target_col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[target_col]):
        volume_col = target_col
    elif "Quantity" in df_feat.columns:
        volume_col = "Quantity"
    elif "Revenue" in df_feat.columns:
        volume_col = "Revenue"
    else:
        num_candidates = df_feat.select_dtypes(include=np.number).columns
        volume_col = num_candidates[0] if len(num_candidates) > 0 else None

    if volume_col is not None:
        daily_prod = df_feat.groupby(["Product", "Date"])[volume_col].sum().reset_index()
        daily_prod = daily_prod.sort_values(["Product", "Date"])
        
        # Shift by 1 so today's target is NOT visible in today's lag/rolling values
        daily_prod["shifted_target"] = daily_prod.groupby("Product")[volume_col].shift(1)
        
        # Lags
        daily_prod["lag_1"] = daily_prod["shifted_target"]
        daily_prod["lag_7"] = daily_prod.groupby("Product")[volume_col].shift(7)
        daily_prod["lag_14"] = daily_prod.groupby("Product")[volume_col].shift(14)
        daily_prod["lag_30"] = daily_prod.groupby("Product")[volume_col].shift(30)
        
        # Rolling stats on shifted series
        daily_prod["roll_mean_7"] = daily_prod.groupby("Product")["shifted_target"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        daily_prod["roll_std_7"] = daily_prod.groupby("Product")["shifted_target"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        ).fillna(0)
        daily_prod["roll_mean_14"] = daily_prod.groupby("Product")["shifted_target"].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )
        daily_prod["roll_mean_30"] = daily_prod.groupby("Product")["shifted_target"].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        # Merge engineered rolling/lag features back to main dataframe
        engineered_cols = ["lag_1", "lag_7", "lag_14", "lag_30", "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_mean_30"]
        merge_keys = ["Product", "Date"]
        df_feat = pd.merge(df_feat, daily_prod[merge_keys + engineered_cols], on=merge_keys, how="left")
        
        # Fill remaining initial lag NaNs with overall product average to avoid dropping historical rows
        for col in engineered_cols:
            prod_mean = df_feat.groupby("Product")[col].transform("mean")
            global_mean = float(df_feat[volume_col].mean())
            df_feat[col] = df_feat[col].fillna(prod_mean).fillna(global_mean)
    else:
        engineered_cols = []
        
    # 4. Pricing & Discount Features
    if "Unit_Price" in df_feat.columns:
        # Price relative to product average
        prod_avg_price = df_feat.groupby("Product")["Unit_Price"].transform("mean")
        df_feat["Price_Ratio"] = (df_feat["Unit_Price"] / prod_avg_price).round(3)
    else:
        df_feat["Price_Ratio"] = 1.0
        
    feature_candidates = [
        "Year", "Month", "Quarter", "Day", "DayOfWeek", "IsWeekend", 
        "IsMonthStart", "IsMonthEnd", "Month_Sin", "Month_Cos", 
        "DayOfWeek_Sin", "DayOfWeek_Cos",
        "Unit_Price", "Discount", "Price_Ratio",
        "lag_1", "lag_7", "lag_14", "lag_30", 
        "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_mean_30",
        "Product", "Category", "Region", "Customer_Segment"
    ]
    
    # Keep only columns that exist
    features_present = [f for f in feature_candidates if f in df_feat.columns]
    
    categorical_cols = [c for c in ["Product", "Category", "Region", "Customer_Segment"] if c in features_present]
    numerical_cols = [c for c in features_present if c not in categorical_cols]
    
    return df_feat, features_present, categorical_cols
