"""
Future sales and demand forecasting module.
Generates multi-horizon time-series projections with confidence bounds and trend analytics.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

from src.utils import get_plotly_layout

def generate_sales_forecast(
    pipeline,
    metadata: dict,
    df_history: pd.DataFrame,
    product: str = "All Products",
    category: str = "All Categories",
    region: str = "All Regions",
    horizon_days: int = 30
) -> dict:
    """
    Project future sales day-by-day across the chosen horizon (7, 14, 30, 60, 90 days).
    Produces daily projections, confidence intervals, and comparison to prior period.
    """
    # 1. Filter historical data
    filtered_hist = df_history.copy()
    if product != "All Products":
        filtered_hist = filtered_hist[filtered_hist["Product"] == product]
    if category != "All Categories":
        filtered_hist = filtered_hist[filtered_hist["Category"] == category]
    if region != "All Regions":
        filtered_hist = filtered_hist[filtered_hist["Region"] == region]
        
    if filtered_hist.empty:
        filtered_hist = df_history.copy()
        
    # Aggregate daily actuals
    daily_actuals = filtered_hist.groupby("Date").agg({
        "Quantity": "sum",
        "Revenue": "sum",
        "Unit_Price": "mean",
        "Discount": "mean"
    }).sort_index().reset_index()
    
    last_date = pd.to_datetime(daily_actuals["Date"].max())
    avg_price = float(daily_actuals["Unit_Price"].mean()) if not daily_actuals.empty else 100.0
    avg_discount = float(daily_actuals["Discount"].mean()) if not daily_actuals.empty else 0.05
    
    # Model RMSE for confidence intervals
    metrics = metadata.get("metrics", [])
    model_rmse = 12.0
    for m in metrics:
        if m.get("Model") == metadata.get("best_model_name"):
            model_rmse = float(m.get("RMSE", 12.0))
            break
            
    # Distinct products present in selection
    unique_prods = filtered_hist["Product"].unique()
    num_prods = max(1, len(unique_prods))
    
    feature_cols = metadata.get("feature_cols", [])
    
    # Rolling state tracking
    recent_quantities = list(daily_actuals["Quantity"].tail(45).values)
    if len(recent_quantities) < 45:
        pad_val = float(daily_actuals["Quantity"].mean())
        recent_quantities = [pad_val] * (45 - len(recent_quantities)) + recent_quantities

    forecast_records = []
    
    for step in range(1, horizon_days + 1):
        fc_date = last_date + timedelta(days=step)
        
        # Calendar features
        dow = fc_date.weekday()
        month = fc_date.month
        is_weekend = 1 if dow in [5, 6] else 0
        
        # Approximate lag values from dynamic buffer
        lag_1 = recent_quantities[-1] / num_prods
        lag_7 = recent_quantities[-7] / num_prods if len(recent_quantities) >= 7 else lag_1
        lag_14 = recent_quantities[-14] / num_prods if len(recent_quantities) >= 14 else lag_1
        lag_30 = recent_quantities[-30] / num_prods if len(recent_quantities) >= 30 else lag_1
        
        roll_mean_7 = np.mean(recent_quantities[-7:]) / num_prods
        roll_std_7 = np.std(recent_quantities[-7:]) / num_prods
        roll_mean_14 = np.mean(recent_quantities[-14:]) / num_prods
        roll_mean_30 = np.mean(recent_quantities[-30:]) / num_prods
        
        row_dict = {
            "Year": fc_date.year,
            "Month": month,
            "Quarter": (month - 1) // 3 + 1,
            "Day": fc_date.day,
            "DayOfWeek": dow,
            "IsWeekend": is_weekend,
            "IsMonthStart": 1 if fc_date.is_month_start else 0,
            "IsMonthEnd": 1 if fc_date.is_month_end else 0,
            "Month_Sin": np.sin(2 * np.pi * month / 12.0),
            "Month_Cos": np.cos(2 * np.pi * month / 12.0),
            "DayOfWeek_Sin": np.sin(2 * np.pi * dow / 7.0),
            "DayOfWeek_Cos": np.cos(2 * np.pi * dow / 7.0),
            "Unit_Price": avg_price,
            "Discount": avg_discount,
            "Price_Ratio": 1.0,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_30": lag_30,
            "roll_mean_7": roll_mean_7,
            "roll_std_7": roll_std_7,
            "roll_mean_14": roll_mean_14,
            "roll_mean_30": roll_mean_30,
            "Product": unique_prods[0] if len(unique_prods) == 1 else "Various",
            "Category": category if category != "All Categories" else "Multiple",
            "Region": region if region != "All Regions" else "Global",
            "Customer_Segment": "Enterprise"
        }
        
        # Build input vector
        inf_df = pd.DataFrame([{c: row_dict.get(c, 0) for c in feature_cols}])
        
        # Base unit projection
        unit_pred = float(pipeline.predict(inf_df)[0])
        total_day_qty = max(1.0, unit_pred * num_prods)
        
        # Add realistic progressive uncertainty interval
        uncertainty_factor = 1.0 + (0.015 * step) # interval widens further into future
        half_ci = model_rmse * np.sqrt(num_prods) * uncertainty_factor
        
        lower_bound = max(0.0, total_day_qty - half_ci)
        upper_bound = total_day_qty + half_ci
        
        day_revenue = total_day_qty * avg_price * (1.0 - avg_discount)
        
        forecast_records.append({
            "Date": fc_date,
            "Predicted_Quantity": round(total_day_qty, 1),
            "Lower_CI": round(lower_bound, 1),
            "Upper_CI": round(upper_bound, 1),
            "Predicted_Revenue": round(day_revenue, 2)
        })
        
        # Push predicted day volume into autoregressive buffer
        recent_quantities.append(total_day_qty)
        if len(recent_quantities) > 90:
            recent_quantities.pop(0)
            
    forecast_df = pd.DataFrame(forecast_records)
    
    # Preceding period comparison (same duration)
    prior_period_actuals = daily_actuals.tail(horizon_days)
    prior_total_qty = float(prior_period_actuals["Quantity"].sum()) if not prior_period_actuals.empty else 1.0
    forecast_total_qty = float(forecast_df["Predicted_Quantity"].sum())
    forecast_total_rev = float(forecast_df["Predicted_Revenue"].sum())
    
    growth_pct = ((forecast_total_qty - prior_total_qty) / max(1.0, prior_total_qty)) * 100.0
    avg_daily_forecast = float(forecast_df["Predicted_Quantity"].mean())
    
    peak_row = forecast_df.loc[forecast_df["Predicted_Quantity"].idxmax()]
    peak_date_str = peak_row["Date"].strftime("%b %d, %Y")
    peak_qty = peak_row["Predicted_Quantity"]
    
    # Construct Interactive Forecast Plotly Figure
    fig = go.Figure()
    
    # 1. Historical Actuals (recent 60 days)
    plot_hist = daily_actuals.tail(60)
    fig.add_trace(go.Scatter(
        x=plot_hist["Date"],
        y=plot_hist["Quantity"],
        mode="lines",
        name="Historical Actuals",
        line=dict(color="#38BDF8", width=2.2),
        hovertemplate="<b>Actual</b><br>Date: %{x|%Y-%m-%d}<br>Units: %{y:,}<extra></extra>"
    ))
    
    # 2. Confidence Interval Ribbon
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
        y=forecast_df["Upper_CI"].tolist() + forecast_df["Lower_CI"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(249, 115, 22, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% Prediction Band"
    ))
    
    # 3. Forecast Trend
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Predicted_Quantity"],
        mode="lines+markers",
        name="Projected Forecast",
        line=dict(color="#F97316", width=2.8, dash="dash"),
        marker=dict(size=4, color="#EA580C"),
        hovertemplate="<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Expected Units: %{y:,}<extra></extra>"
    ))
    
    # Connect last actual to first forecast for seamless continuity
    last_actual_pt = plot_hist.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_actual_pt["Date"], forecast_df.iloc[0]["Date"]],
        y=[last_actual_pt["Quantity"], forecast_df.iloc[0]["Predicted_Quantity"]],
        mode="lines",
        line=dict(color="#F97316", width=1.8, dash="dot"),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    layout = get_plotly_layout()
    layout.update(
        title=f"Sales Demand Forecast ({horizon_days}-Day Horizon)",
        xaxis_title="Timeline",
        yaxis_title="Units Sold",
        hovermode="x unified",
        height=450
    )
    fig.update_layout(layout)
    
    return {
        "forecast_df": forecast_df,
        "forecast_total_qty": forecast_total_qty,
        "forecast_total_rev": forecast_total_rev,
        "avg_daily_forecast": avg_daily_forecast,
        "growth_pct": round(growth_pct, 1),
        "peak_date_str": peak_date_str,
        "peak_qty": peak_qty,
        "fig": fig
    }
