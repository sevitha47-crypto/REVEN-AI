"""
Interactive single and batch sales prediction engine.
Uses trained ML model pipelines to predict demand quantities and expected revenues.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def predict_single_sales(
    pipeline,
    metadata: dict,
    df_history: pd.DataFrame,
    product: str,
    category: str,
    region: str,
    customer_segment: str,
    target_date: str,
    unit_price: float,
    discount: float
) -> dict:
    """
    Generate inference for a single scenario using the trained model pipeline.
    Synthesizes the exact calendar and historical lag/rolling state for that product.
    """
    dt = pd.to_datetime(target_date)
    
    # 1. Product baseline statistics from history
    prod_history = df_history[df_history["Product"] == product]
    if not prod_history.empty:
        hist_qty_mean = float(prod_history["Quantity"].mean())
        hist_qty_std = float(prod_history["Quantity"].std()) if len(prod_history) > 1 else 1.0
        hist_price_avg = float(prod_history["Unit_Price"].mean())
        # Most recent lag proxies
        recent_daily = prod_history.groupby("Date")["Quantity"].sum().sort_index()
        lag_1 = float(recent_daily.iloc[-1]) if len(recent_daily) >= 1 else hist_qty_mean
        lag_7 = float(recent_daily.iloc[-7]) if len(recent_daily) >= 7 else hist_qty_mean
        lag_14 = float(recent_daily.iloc[-14]) if len(recent_daily) >= 14 else hist_qty_mean
        lag_30 = float(recent_daily.iloc[-30]) if len(recent_daily) >= 30 else hist_qty_mean
        roll_mean_7 = float(recent_daily.tail(7).mean()) if len(recent_daily) >= 1 else hist_qty_mean
        roll_std_7 = float(recent_daily.tail(7).std()) if len(recent_daily) >= 2 else 0.0
        roll_mean_14 = float(recent_daily.tail(14).mean()) if len(recent_daily) >= 1 else hist_qty_mean
        roll_mean_30 = float(recent_daily.tail(30).mean()) if len(recent_daily) >= 1 else hist_qty_mean
    else:
        hist_qty_mean = float(df_history["Quantity"].mean())
        hist_qty_std = float(df_history["Quantity"].std())
        hist_price_avg = unit_price
        lag_1 = lag_7 = lag_14 = lag_30 = roll_mean_7 = roll_mean_14 = roll_mean_30 = hist_qty_mean
        roll_std_7 = 0.0

    # 2. Build single-row feature dataframe matching the training schema
    year = dt.year
    month = dt.month
    quarter = dt.quarter
    day = dt.day
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek in [5, 6] else 0
    is_m_start = 1 if dt.is_month_start else 0
    is_m_end = 1 if dt.is_month_end else 0
    
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7.0)
    
    price_ratio = round(unit_price / max(0.01, hist_price_avg), 3)

    feature_cols = metadata.get("feature_cols", [])
    
    record = {
        "Year": year,
        "Month": month,
        "Quarter": quarter,
        "Day": day,
        "DayOfWeek": dayofweek,
        "IsWeekend": is_weekend,
        "IsMonthStart": is_m_start,
        "IsMonthEnd": is_m_end,
        "Month_Sin": month_sin,
        "Month_Cos": month_cos,
        "DayOfWeek_Sin": dow_sin,
        "DayOfWeek_Cos": dow_cos,
        "Unit_Price": unit_price,
        "Discount": discount,
        "Price_Ratio": price_ratio,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_30": lag_30,
        "roll_mean_7": roll_mean_7,
        "roll_std_7": roll_std_7,
        "roll_mean_14": roll_mean_14,
        "roll_mean_30": roll_mean_30,
        "Product": product,
        "Category": category,
        "Region": region,
        "Customer_Segment": customer_segment
    }
    
    # Filter only columns the model expects
    input_df = pd.DataFrame([{c: record.get(c, 0) for c in feature_cols}])
    
    # Run prediction
    raw_pred = pipeline.predict(input_df)[0]
    predicted_qty = max(1, int(round(raw_pred)))
    expected_revenue = round(predicted_qty * unit_price * (1.0 - discount), 2)
    
    # Demand Classification relative to product historical baseline
    diff_from_mean = predicted_qty - hist_qty_mean
    pct_from_mean = (diff_from_mean / max(1.0, hist_qty_mean)) * 100.0
    
    if diff_from_mean >= 0.45 * hist_qty_std or pct_from_mean >= 18.0:
        demand_level = "High Demand"
        demand_badge_class = "badge-high"
    elif diff_from_mean <= -0.45 * hist_qty_std or pct_from_mean <= -18.0:
        demand_level = "Low Demand"
        demand_badge_class = "badge-low"
    else:
        demand_level = "Medium Demand"
        demand_badge_class = "badge-medium"
        
    context = {
        "historical_avg_qty": round(hist_qty_mean, 1),
        "historical_std_qty": round(hist_qty_std, 1),
        "pct_vs_historical": round(pct_from_mean, 1),
        "unit_diff": round(diff_from_mean, 1),
        "is_weekend": bool(is_weekend),
        "price_ratio": price_ratio,
        "day_name": dt.strftime("%A"),
        "date_str": dt.strftime("%B %d, %Y")
    }
    
    return {
        "predicted_quantity": predicted_qty,
        "expected_revenue": expected_revenue,
        "demand_level": demand_level,
        "demand_badge_class": demand_badge_class,
        "context": context,
        "model_used": metadata.get("best_model_name", "Trained ML Model")
    }


def predict_classification_single(
    pipeline,
    metadata: dict,
    df_history: pd.DataFrame,
    product: str,
    category: str,
    region: str,
    customer_segment: str,
    target_date: str,
    unit_price: float,
    discount: float
) -> dict:
    """
    Run inference on a single operational scenario using the trained classification champion pipeline.
    Returns predicted class, confidence level, full class probability breakdown, and strategic action plan.
    """
    dt = pd.to_datetime(target_date)
    
    # 1. Product baseline statistics from history
    prod_history = df_history[df_history["Product"] == product]
    if not prod_history.empty:
        hist_qty_mean = float(prod_history["Quantity"].mean())
        hist_qty_std = float(prod_history["Quantity"].std()) if len(prod_history) > 1 else 1.0
        hist_price_avg = float(prod_history["Unit_Price"].mean())
        recent_daily = prod_history.groupby("Date")["Quantity"].sum().sort_index()
        lag_1 = float(recent_daily.iloc[-1]) if len(recent_daily) >= 1 else hist_qty_mean
        lag_7 = float(recent_daily.iloc[-7]) if len(recent_daily) >= 7 else hist_qty_mean
        lag_14 = float(recent_daily.iloc[-14]) if len(recent_daily) >= 14 else hist_qty_mean
        lag_30 = float(recent_daily.iloc[-30]) if len(recent_daily) >= 30 else hist_qty_mean
        roll_mean_7 = float(recent_daily.tail(7).mean()) if len(recent_daily) >= 1 else hist_qty_mean
        roll_std_7 = float(recent_daily.tail(7).std()) if len(recent_daily) >= 2 else 0.0
        roll_mean_14 = float(recent_daily.tail(14).mean()) if len(recent_daily) >= 1 else hist_qty_mean
        roll_mean_30 = float(recent_daily.tail(30).mean()) if len(recent_daily) >= 1 else hist_qty_mean
    else:
        hist_qty_mean = float(df_history["Quantity"].mean())
        hist_qty_std = float(df_history["Quantity"].std())
        hist_price_avg = unit_price
        lag_1 = lag_7 = lag_14 = lag_30 = roll_mean_7 = roll_mean_14 = roll_mean_30 = hist_qty_mean
        roll_std_7 = 0.0

    # 2. Build single-row feature dataframe
    year = dt.year
    month = dt.month
    quarter = dt.quarter
    day = dt.day
    dayofweek = dt.weekday()
    is_weekend = 1 if dayofweek in [5, 6] else 0
    is_m_start = 1 if dt.is_month_start else 0
    is_m_end = 1 if dt.is_month_end else 0
    
    price_ratio = round(unit_price / max(0.01, hist_price_avg), 2)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7.0)
    
    feature_cols = metadata.get("feature_cols", [])
    
    feature_dict = {
        "Year": year,
        "Month": month,
        "Quarter": quarter,
        "Day": day,
        "DayOfWeek": dayofweek,
        "IsWeekend": is_weekend,
        "IsMonthStart": is_m_start,
        "IsMonthEnd": is_m_end,
        "Month_Sin": month_sin,
        "Month_Cos": month_cos,
        "DayOfWeek_Sin": dow_sin,
        "DayOfWeek_Cos": dow_cos,
        "Unit_Price": unit_price,
        "Discount": discount,
        "Price_Ratio": price_ratio,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_30": lag_30,
        "roll_mean_7": roll_mean_7,
        "roll_std_7": roll_std_7,
        "roll_mean_14": roll_mean_14,
        "roll_mean_30": roll_mean_30,
        "Product": product,
        "Category": category,
        "Region": region,
        "Customer_Segment": customer_segment
    }
    
    sample_df = pd.DataFrame([{col: feature_dict.get(col, 0) for col in feature_cols}])
    
    # 3. Model Inference
    predicted_class = str(pipeline.predict(sample_df)[0])
    
    class_probabilities = {}
    confidence = 90.0
    
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(sample_df)[0]
            classes = pipeline.classes_
            for cls, prob in zip(classes, probs):
                class_probabilities[str(cls)] = round(float(prob) * 100, 1)
            confidence = class_probabilities.get(predicted_class, round(float(np.max(probs)) * 100, 1))
        except Exception:
            pass

    # 4. Strategic action plan based on classification outcome
    target_name = metadata.get("target_col", "Demand Velocity")
    if "high" in predicted_class.lower():
        badge_style = "badge-high"
        strategy = f"High priority tier for {product}. Expedite supply chain buffer, prioritize distribution slots, and minimize discount depth."
    elif "low" in predicted_class.lower():
        badge_style = "badge-low"
        strategy = f"Lower velocity tier. Restrict forward inventory commitments, explore bundling with high-velocity items, and evaluate promotional clearance."
    else:
        badge_style = "badge-medium"
        strategy = f"Standard stable velocity. Maintain nominal safety stock, monitor weekly reorder points, and maintain standard price margins."

    return {
        "predicted_class": predicted_class,
        "confidence_pct": confidence,
        "class_probabilities": class_probabilities,
        "badge_style": badge_style,
        "target_name": target_name,
        "strategy": strategy,
        "model_used": metadata.get("best_model_name", "Champion Classifier")
    }

