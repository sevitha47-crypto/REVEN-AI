"""
Inventory risk and stock intelligence module.
Calculates days of inventory remaining (DOIR), demand pressure, stockout risks,
and recommended reorder units using statistical safety stock formulas.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import get_plotly_layout

def analyze_inventory_risk(
    demand_df: pd.DataFrame,
    df_history: pd.DataFrame,
    horizon_days: int = 30,
    target_buffer_days: int = 45
) -> dict:
    """
    Perform transparent inventory risk calculations.
    Evaluates current inventory against predicted demand velocity.
    """
    # Latest stock values from dataset
    latest_stock = df_history.groupby("Product")["Current_Stock"].last().to_dict()
    
    inventory_records = []
    
    for _, row in demand_df.iterrows():
        prod = row["Product"]
        stock_on_hand = latest_stock.get(prod, 150)
        daily_pred = max(0.1, row["Predicted_Daily_Demand"])
        hist_daily = max(0.1, row["Historical_Daily_Avg"])
        
        # Days of Inventory Remaining (DOIR) based on predicted velocity
        doir = stock_on_hand / daily_pred
        
        # Predicted Demand over the forecast horizon
        horizon_demand = daily_pred * horizon_days
        
        # Deficit or Surplus at horizon end
        projected_balance = stock_on_hand - horizon_demand
        
        # Min & Max inventory limits for each product
        min_safety_stock = int(np.ceil(daily_pred * 14))
        max_capacity_stock = int(np.ceil(daily_pred * 90))
        optimal_target_stock = int(np.ceil(daily_pred * target_buffer_days))
        reorder_units = max(0, int(np.ceil(optimal_target_stock - stock_on_hand)))
        
        # Risk Classification
        if stock_on_hand < min_safety_stock or doir < horizon_days:
            risk_status = "HIGH RISK"
            risk_badge = "badge-high"
            risk_desc = f"Stock below minimum safety threshold ({min_safety_stock} units). Stockout within {int(round(doir))} days."
        elif stock_on_hand > max_capacity_stock:
            risk_status = "MEDIUM RISK"
            risk_badge = "badge-medium"
            risk_desc = f"Stock exceeds maximum inventory capacity ({max_capacity_stock} units). Risk of excess carrying cost."
        elif doir < (horizon_days * 1.5):
            risk_status = "MEDIUM RISK"
            risk_badge = "badge-medium"
            risk_desc = f"Inventory covers {int(round(doir))} days. Approaching reorder safety threshold."
        else:
            risk_status = "LOW RISK"
            risk_badge = "badge-low"
            risk_desc = f"Stable stock position ({int(round(doir))} days of coverage between min and max bounds)."
            
        inventory_records.append({
            "Product": prod,
            "Category": row["Category"],
            "Current_Stock": stock_on_hand,
            "Min_Safety_Stock": min_safety_stock,
            "Max_Capacity_Stock": max_capacity_stock,
            "Predicted_Daily_Demand": daily_pred,
            "Horizon_Demand": round(horizon_demand, 1),
            "DOIR_Days": round(doir, 1),
            "Projected_Ending_Stock": round(projected_balance, 1),
            "Recommended_Reorder_Units": reorder_units,
            "Risk_Status": risk_status,
            "Risk_Badge": risk_badge,
            "Risk_Description": risk_desc
        })
        
    inv_df = pd.DataFrame(inventory_records).sort_values("DOIR_Days", ascending=True).reset_index(drop=True)
    
    # Aggregates
    high_risk_count = int((inv_df["Risk_Status"] == "HIGH RISK").sum())
    med_risk_count = int((inv_df["Risk_Status"] == "MEDIUM RISK").sum())
    low_risk_count = int((inv_df["Risk_Status"] == "LOW RISK").sum())
    total_reorder_units = int(inv_df["Recommended_Reorder_Units"].sum())
    
    # Inventory Coverage Chart
    colors = {"HIGH RISK": "#EF4444", "MEDIUM RISK": "#F59E0B", "LOW RISK": "#10B981"}
    fig_coverage = px.bar(
        inv_df,
        x="DOIR_Days",
        y="Product",
        orientation="h",
        color="Risk_Status",
        color_discrete_map=colors,
        text="DOIR_Days",
        title=f"Days of Inventory Remaining (DOIR) vs. {horizon_days}-Day Horizon"
    )
    fig_coverage.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
    # Add vertical line for horizon threshold
    fig_coverage.add_vline(
        x=horizon_days,
        line_width=2,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Horizon Threshold ({horizon_days}d)",
        annotation_position="top right",
        annotation_font_color="#FCA5A5"
    )
    
    layout = get_plotly_layout()
    layout.update(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Days of Inventory Remaining",
        yaxis_title="Product",
        height=max(400, len(inv_df) * 32)
    )
    fig_coverage.update_layout(layout)
    
    return {
        "inv_df": inv_df,
        "high_risk_count": high_risk_count,
        "med_risk_count": med_risk_count,
        "low_risk_count": low_risk_count,
        "total_reorder_units": total_reorder_units,
        "fig_coverage": fig_coverage
    }
