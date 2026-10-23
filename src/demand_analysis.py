"""
Demand intelligence module.
Classifies future product demand into High, Medium, and Low categories based on
predicted vs historical demand velocity and statistical thresholds.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import get_plotly_layout

def analyze_demand_landscape(
    pipeline,
    metadata: dict,
    df_history: pd.DataFrame,
    horizon_days: int = 30
) -> dict:
    """
    Compute demand intelligence metrics across all products in the catalog.
    Compares baseline historical run-rate to projected future demand.
    """
    feature_cols = metadata.get("feature_cols", [])
    df_history = df_history.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_history["Date"]):
        df_history["Date"] = pd.to_datetime(df_history["Date"])
    
    # 1. Historical baseline metrics per product (recent 90 days or all history)
    recent_cutoff = df_history["Date"].max() - pd.Timedelta(days=90)
    df_recent = df_history[df_history["Date"] >= recent_cutoff]
    if df_recent.empty:
        df_recent = df_history.copy()
        
    hist_stats = df_recent.groupby("Product").agg({
        "Quantity": ["mean", "sum", "std"],
        "Revenue": "sum",
        "Unit_Price": "mean",
        "Category": "first",
        "Region": "first"
    }).reset_index()
    
    hist_stats.columns = ["Product", "Hist_Daily_Mean", "Hist_Total_Qty", "Hist_Qty_Std", "Hist_Revenue", "Avg_Price", "Category", "Sample_Region"]
    hist_stats["Hist_Daily_Mean"] = hist_stats["Hist_Daily_Mean"].fillna(1.0).round(2)
    hist_stats["Hist_Qty_Std"] = hist_stats["Hist_Qty_Std"].fillna(0.5).round(2)
    
    # Total history revenue for share calculation
    total_rev = hist_stats["Hist_Revenue"].sum()
    hist_stats["Revenue_Share_Pct"] = ((hist_stats["Hist_Revenue"] / max(1.0, total_rev)) * 100.0).round(1)

    # 2. Estimate future demand using the trained model across future horizon
    # Construct standard representative scenario for each product
    future_date = df_history["Date"].max() + pd.Timedelta(days=horizon_days // 2)
    dow = future_date.weekday()
    month = future_date.month
    
    predictions = []
    for _, row in hist_stats.iterrows():
        input_record = {
            "Year": future_date.year,
            "Month": month,
            "Quarter": (month - 1) // 3 + 1,
            "Day": future_date.day,
            "DayOfWeek": dow,
            "IsWeekend": 1 if dow in [5, 6] else 0,
            "IsMonthStart": 0,
            "IsMonthEnd": 0,
            "Month_Sin": np.sin(2 * np.pi * month / 12.0),
            "Month_Cos": np.cos(2 * np.pi * month / 12.0),
            "DayOfWeek_Sin": np.sin(2 * np.pi * dow / 7.0),
            "DayOfWeek_Cos": np.cos(2 * np.pi * dow / 7.0),
            "Unit_Price": row["Avg_Price"],
            "Discount": 0.05,
            "Price_Ratio": 1.0,
            "lag_1": row["Hist_Daily_Mean"],
            "lag_7": row["Hist_Daily_Mean"],
            "lag_14": row["Hist_Daily_Mean"],
            "lag_30": row["Hist_Daily_Mean"],
            "roll_mean_7": row["Hist_Daily_Mean"],
            "roll_std_7": row["Hist_Qty_Std"],
            "roll_mean_14": row["Hist_Daily_Mean"],
            "roll_mean_30": row["Hist_Daily_Mean"],
            "Product": row["Product"],
            "Category": row["Category"],
            "Region": row["Sample_Region"],
            "Customer_Segment": "Enterprise"
        }
        
        sample_df = pd.DataFrame([{c: input_record.get(c, 0) for c in feature_cols}])
        pred_daily = float(pipeline.predict(sample_df)[0])
        pred_daily = max(1.0, round(pred_daily, 2))
        pred_horizon = round(pred_daily * horizon_days, 1)
        
        # Calculate demand velocity change
        hist_daily = max(0.1, row["Hist_Daily_Mean"])
        change_pct = ((pred_daily - hist_daily) / hist_daily) * 100.0
        
        # Statistical Classification
        if change_pct >= 15.0 or (pred_daily - hist_daily) >= 0.5 * row["Hist_Qty_Std"]:
            level = "High Demand"
            badge = "badge-high"
        elif change_pct <= -15.0 or (pred_daily - hist_daily) <= -0.5 * row["Hist_Qty_Std"]:
            level = "Low Demand"
            badge = "badge-low"
        else:
            level = "Medium Demand"
            badge = "badge-medium"
            
        predictions.append({
            "Product": row["Product"],
            "Category": row["Category"],
            "Historical_Daily_Avg": hist_daily,
            "Predicted_Daily_Demand": pred_daily,
            "Predicted_Horizon_Demand": pred_horizon,
            "Demand_Change_Pct": round(change_pct, 1),
            "Demand_Level": level,
            "Badge": badge,
            "Revenue_Share_Pct": row["Revenue_Share_Pct"]
        })
        
    demand_df = pd.DataFrame(predictions).sort_values("Demand_Change_Pct", ascending=False).reset_index(drop=True)
    
    # Summary Counts
    level_counts = demand_df["Demand_Level"].value_counts().to_dict()
    high_count = level_counts.get("High Demand", 0)
    med_count = level_counts.get("Medium Demand", 0)
    low_count = level_counts.get("Low Demand", 0)
    
    # Visualization: Demand Distribution
    colors = {"High Demand": "#EF4444", "Medium Demand": "#F59E0B", "Low Demand": "#10B981"}
    
    fig_bar = px.bar(
        demand_df,
        x="Demand_Change_Pct",
        y="Product",
        orientation="h",
        color="Demand_Level",
        color_discrete_map=colors,
        text="Demand_Change_Pct",
        title=f"Product Demand Velocity vs Baseline ({horizon_days}-Day Horizon)"
    )
    fig_bar.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
    layout = get_plotly_layout()
    layout.update(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Predicted Demand Change vs. Baseline (%)",
        yaxis_title="Product",
        height=max(400, len(demand_df) * 32)
    )
    fig_bar.update_layout(layout)
    
    return {
        "demand_df": demand_df,
        "high_count": high_count,
        "med_count": med_count,
        "low_count": low_count,
        "fig_bar": fig_bar
    }
