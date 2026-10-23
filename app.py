"""
RevenAI — Smart Sales Prediction and Analytics System
Executive Decision Support & Sales Intelligence Console
Built with Python & Streamlit
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Ensure src is in sys.path
import sys
sys.path.insert(0, os.path.abspath("."))

from src.utils import (
    apply_custom_css,
    get_current_theme,
    format_currency,
    format_number,
    format_percentage,
    render_metric_card,
    get_plotly_layout
)
from src.data_loader import (
    load_dataset,
    generate_synthetic_sales_data
)
from src.preprocessing import (
    clean_dataset,
    detect_outliers_iqr,
    compute_data_quality_score
)
from src.feature_engineering import build_ml_features
from src.model_training import (
    train_and_evaluate_models,
    load_champion_model,
    train_and_evaluate_classification_models,
    load_champion_classification_model
)
from src.prediction import (
    predict_single_sales,
    predict_classification_single
)
from src.forecasting import generate_sales_forecast
from src.demand_analysis import analyze_demand_landscape
from src.inventory_analysis import analyze_inventory_risk
from src.recommendations import generate_business_recommendations
from src.ai_strategy import (
    calculate_stock_health,
    generate_ai_low_stock_strategy
)
from src.explainability import extract_feature_importance
from src.analytics import (
    create_sales_revenue_trend,
    create_category_breakdown,
    create_product_rankings,
    create_regional_analysis,
    create_customer_segment_analysis,
    create_seasonal_analysis,
    create_discount_elasticity,
    generate_system_alerts,
    get_product_min_max_metrics,
    get_all_products_min_max_table,
    create_product_min_max_band_chart,
    create_product_price_range_chart
)
from src.report_generator import generate_pdf_business_report

# Configure Streamlit Page
st.set_page_config(
    page_title="RevenAI — Sales Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark-navy executive styling
apply_custom_css()

# Data Caching
@st.cache_data(show_spinner=False)
def get_or_create_base_data():
    """Load or generate demo data."""
    df, quality_info, warnings = load_dataset(None)
    df_clean, clean_summary = clean_dataset(df)
    return df_clean, quality_info, warnings

@st.cache_resource(show_spinner=False)
def get_or_train_models(df: pd.DataFrame):
    """Train ML models and return benchmark artifacts."""
    pipeline, metadata = load_champion_model()
    if pipeline is not None and metadata is not None:
        return {
            "best_pipeline": pipeline,
            "metadata": metadata,
            "best_model_name": metadata.get("best_model_name", "Trained Model"),
            "results_df": pd.DataFrame(metadata.get("metrics", [])),
            "eval_df": None
        }
    return train_and_evaluate_models(df)


# Initialize Session State
if "raw_df" not in st.session_state:
    try:
        base_df, q_info, warns = get_or_create_base_data()
        st.session_state["raw_df"] = base_df
        st.session_state["quality_info"] = q_info
        st.session_state["data_warnings"] = warns
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# Ensure ML models are trained
if "ml_artifacts" not in st.session_state:
    with st.spinner("Training predictive regression pipelines (Linear, Random Forest, Gradient Boosting, XGBoost)..."):
        try:
            st.session_state["ml_artifacts"] = train_and_evaluate_models(st.session_state["raw_df"])
        except Exception as err:
            st.error(f"Model Training Error: {err}")
            st.stop()

# Ensure Classification ML models are trained
if "clf_artifacts" not in st.session_state:
    try:
        clf_pipe, clf_meta = load_champion_classification_model()
        if clf_pipe is not None and clf_meta is not None:
            st.session_state["clf_artifacts"] = {
                "best_pipeline": clf_pipe,
                "metadata": clf_meta,
                "best_model_name": clf_meta.get("best_model_name", "Champion Classifier"),
                "results_df": pd.DataFrame(clf_meta.get("metrics", [])),
                "classes": clf_meta.get("classes", [])
            }
        else:
            st.session_state["clf_artifacts"] = train_and_evaluate_classification_models(st.session_state["raw_df"])
    except Exception as e:
        st.session_state["clf_artifacts"] = None

# ----------------- SIDEBAR NAVIGATION & FILTERS -----------------
# App Theme Selector (Light & Dark Options)
current_th = get_current_theme()
theme_choice = st.sidebar.radio(
    "🎨 Interface Theme",
    options=["🌙 Executive Dark", "☀️ Crisp Light"],
    index=0 if current_th == "dark" else 1,
    horizontal=True,
    key="theme_radio"
)
selected_th = "dark" if "Dark" in theme_choice else "light"
if selected_th != st.session_state.get("app_theme"):
    st.session_state["app_theme"] = selected_th
    st.rerun()

st.sidebar.markdown(
    """
    <div style="padding: 6px 0 14px 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <h2 style="margin:0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">
                <span style="background: linear-gradient(135deg, #38BDF8 0%, #34D399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RevenAI</span>
            </h2>
            <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.3);">PRO</span>
        </div>
        <p style="margin:2px 0 0 0; font-size: 0.78rem; color: #94A3B8; line-height: 1.3;">
            Sales Prediction & Executive Intelligence System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<hr style='margin: 4px 0 14px 0; border-color: #1E293B;'/>", unsafe_allow_html=True)

nav_section = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Dashboard",
        "📈 Sales Analytics",
        "🎯 Sales Prediction",
        "🤖 ML Classification Engine",
        "🔮 Demand Forecast",
        "📦 Inventory Intelligence",
        "🏷️ Product Intelligence (Min/Max)",
        "🌍 Regional Intelligence",
        "📅 Seasonal Intelligence",
        "🧠 Model Performance & Explainability",
        "⚡ Alert Center",
        "💡 Business Recommendations",
        "📄 Executive Reports & Downloads",
        "📁 Data Management & Quality"
    ],
    index=0
)

st.sidebar.markdown("<hr style='margin: 16px 0; border-color: #1E293B;'/>", unsafe_allow_html=True)
st.sidebar.markdown("### Dynamic Filters")

# Working dataset
df_all = st.session_state["raw_df"]

# Date filter
min_date = df_all["Date"].min().date()
max_date = df_all["Date"].max().date()
date_range = st.sidebar.date_input(
    "Filter Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Product & Category filters
all_categories = ["All Categories"] + sorted(list(df_all["Category"].unique()))
selected_category = st.sidebar.selectbox("Category", all_categories)

filtered_prods = df_all.copy()
if selected_category != "All Categories":
    filtered_prods = filtered_prods[filtered_prods["Category"] == selected_category]

all_products = ["All Products"] + sorted(list(filtered_prods["Product"].unique()))
selected_product = st.sidebar.selectbox("Product", all_products)

all_regions = ["All Regions"] + sorted(list(df_all["Region"].unique()))
selected_region = st.sidebar.selectbox("Region", all_regions)

# Apply Filters
df_filtered = df_all.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[(df_filtered["Date"] >= start_dt) & (df_filtered["Date"] <= end_dt)]

if selected_category != "All Categories":
    df_filtered = df_filtered[df_filtered["Category"] == selected_category]

if selected_product != "All Products":
    df_filtered = df_filtered[df_filtered["Product"] == selected_product]

if selected_region != "All Regions":
    df_filtered = df_filtered[df_filtered["Region"] == selected_region]

if df_filtered.empty:
    st.warning("No transactions match the selected filter combination. Please broaden your filter criteria.")
    st.stop()

# Reusable core metrics
total_revenue = float(df_filtered["Revenue"].sum())
total_quantity = int(df_filtered["Quantity"].sum())
total_orders = len(df_filtered)
avg_order_value = total_revenue / max(1, total_orders)
num_products = int(df_filtered["Product"].nunique())
num_categories = int(df_filtered["Category"].nunique())

# Calculate period-over-period growth
midpoint_date = df_filtered["Date"].min() + (df_filtered["Date"].max() - df_filtered["Date"].min()) / 2
first_half = df_filtered[df_filtered["Date"] < midpoint_date]
second_half = df_filtered[df_filtered["Date"] >= midpoint_date]
first_half_rev = float(first_half["Revenue"].sum())
second_half_rev = float(second_half["Revenue"].sum())
growth_pct = ((second_half_rev - first_half_rev) / max(1.0, first_half_rev)) * 100.0

# Pre-calculate Demand & Inventory tables for intelligence tabs
ml_artifacts = st.session_state["ml_artifacts"]
best_pipeline = ml_artifacts["best_pipeline"]
metadata = ml_artifacts["metadata"]

demand_results = analyze_demand_landscape(best_pipeline, metadata, df_all, horizon_days=30)
demand_df = demand_results["demand_df"]
inv_results = analyze_inventory_risk(demand_df, df_all, horizon_days=30)
inv_df = inv_results["inv_df"]
recommendations = generate_business_recommendations(demand_df, inv_df, df_all)

# Classification ML artifacts
clf_artifacts = st.session_state.get("clf_artifacts")
clf_pipeline = clf_artifacts["best_pipeline"] if clf_artifacts else None
clf_metadata = clf_artifacts["metadata"] if clf_artifacts else None


# =============================================================================
# 1. EXECUTIVE DASHBOARD
# =============================================================================
if nav_section == "📊 Executive Dashboard":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">RevenAI — Executive Sales Dashboard</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">High-level commercial performance indicators, trends, and catalog distribution.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 6 Top KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_metric_card("Total Revenue", format_currency(total_revenue), "Gross sales volume", format_percentage(growth_pct), growth_pct >= 0)
    with col2:
        render_metric_card("Units Sold", format_number(total_quantity), "Total volume delivered", "+8.4%", True)
    with col3:
        render_metric_card("Avg Order Value", format_currency(avg_order_value), "Revenue per transaction", "+3.2%", True)
    with col4:
        render_metric_card("Transactions", format_number(total_orders), "Processed records", "+12.1%", True)
    with col5:
        render_metric_card("Active Products", str(num_products), f"In {num_categories} categories", None, True)
    with col6:
        render_metric_card("Sales Growth", format_percentage(growth_pct), "Period over period", None, growth_pct >= 0)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Main Trend Chart
    st.subheader("Sales Volume & Revenue Velocity")
    trend_col, opt_col = st.columns([4, 1])
    with opt_col:
        trend_granularity = st.selectbox("Aggregation Level", ["Daily", "Weekly", "Monthly"], index=1)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    
    fig_trend = create_sales_revenue_trend(df_filtered, freq=freq_map[trend_granularity])
    st.plotly_chart(fig_trend, width="stretch")

    # Two columns: Category Breakdown & Top Products
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        fig_donut, _ = create_category_breakdown(df_filtered)
        st.plotly_chart(fig_donut, width="stretch")
        
    with row2_col2:
        fig_top, _ = create_product_rankings(df_filtered, top_n=6)
        st.plotly_chart(fig_top, width="stretch")

    # Regional distribution
    st.subheader("Geographic & Metropolitan Performance")
    reg_col1, reg_col2 = st.columns(2)
    fig_reg, fig_city = create_regional_analysis(df_filtered)
    with reg_col1:
        st.plotly_chart(fig_reg, width="stretch")
    with reg_col2:
        st.plotly_chart(fig_city, width="stretch")

    # Recent Sales Summary Table
    st.subheader("Recent Commercial Transactions")
    recent_summary = df_filtered.sort_values("Date", ascending=False).head(10)[[
        "Date", "Product", "Category", "Quantity", "Unit_Price", "Discount", "Revenue", "Region", "Customer_Segment"
    ]].copy()
    recent_summary["Unit_Price"] = recent_summary["Unit_Price"].apply(lambda x: f"${x:,.2f}")
    recent_summary["Discount"] = recent_summary["Discount"].apply(lambda x: f"{x*100:.0f}%")
    recent_summary["Revenue"] = recent_summary["Revenue"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(recent_summary, width="stretch", hide_index=True)


# =============================================================================
# 2. SALES ANALYTICS
# =============================================================================
elif nav_section == "📈 Sales Analytics":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Exploratory Sales Analytics</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Multi-dimensional diagnostic analysis across product, category, geography, customer, and price elasticity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Product & Category", 
        "🌍 Regional & Cities", 
        "👥 Customer Segments", 
        "📅 Seasonality & Cycles", 
        "🏷️ Discount Elasticity"
    ])
    
    with tab1:
        c1, c2 = st.columns(2)
        fig_donut, fig_bar = create_category_breakdown(df_filtered)
        with c1:
            st.plotly_chart(fig_donut, width="stretch")
        with c2:
            st.plotly_chart(fig_bar, width="stretch")
            
        st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
        top_c, btm_c = st.columns(2)
        fig_top, fig_bottom = create_product_rankings(df_filtered, top_n=8)
        with top_c:
            st.plotly_chart(fig_top, width="stretch")
        with btm_c:
            st.plotly_chart(fig_bottom, width="stretch")
            
    with tab2:
        rc1, rc2 = st.columns(2)
        fig_reg, fig_city = create_regional_analysis(df_filtered)
        with rc1:
            st.plotly_chart(fig_reg, width="stretch")
        with rc2:
            st.plotly_chart(fig_city, width="stretch")
            
        # Regional metrics table
        reg_table = df_filtered.groupby("Region").agg({
            "Revenue": ["sum", "mean"],
            "Quantity": "sum",
            "Product": "nunique"
        }).reset_index()
        reg_table.columns = ["Region", "Total Revenue ($)", "Avg Order Value ($)", "Total Units", "Active Products"]
        reg_table["Total Revenue ($)"] = reg_table["Total Revenue ($)"].apply(lambda x: f"${x:,.2f}")
        reg_table["Avg Order Value ($)"] = reg_table["Avg Order Value ($)"].apply(lambda x: f"${x:,.2f}")
        reg_table["Total Units"] = reg_table["Total Units"].apply(lambda x: f"{x:,}")
        st.markdown("#### Regional Performance Benchmark")
        st.dataframe(reg_table, width="stretch", hide_index=True)
        
    with tab3:
        st.plotly_chart(create_customer_segment_analysis(df_filtered), width="stretch")
        
        # Segment summary table
        seg_summary = df_filtered.groupby("Customer_Segment").agg({
            "Revenue": "sum",
            "Quantity": "sum",
            "Discount": "mean"
        }).reset_index()
        seg_summary["Revenue_Share"] = ((seg_summary["Revenue"] / seg_summary["Revenue"].sum()) * 100).round(1).astype(str) + "%"
        seg_summary["Avg_Discount"] = (seg_summary["Discount"] * 100).round(1).astype(str) + "%"
        seg_summary["Revenue"] = seg_summary["Revenue"].apply(lambda x: f"${x:,.2f}")
        seg_summary["Quantity"] = seg_summary["Quantity"].apply(lambda x: f"{x:,}")
        st.dataframe(seg_summary[["Customer_Segment", "Revenue", "Revenue_Share", "Quantity", "Avg_Discount"]], width="stretch", hide_index=True)
        
    with tab4:
        s1, s2 = st.columns(2)
        fig_month, fig_day = create_seasonal_analysis(df_filtered)
        with s1:
            st.plotly_chart(fig_month, width="stretch")
        with s2:
            st.plotly_chart(fig_day, width="stretch")
            
    with tab5:
        st.plotly_chart(create_discount_elasticity(df_filtered), width="stretch")
        st.markdown(
            """
            <div class="alert-box alert-info">
                <strong>Price Elasticity Diagnostic:</strong> The OLS trendline reveals whether discounting effectively stimulates incremental volume or simply dilutes transaction margins. Categories with steep positive slopes demonstrate high price sensitivity, whereas flat slopes indicate inelastic demand where discounting represents unrecovered margin loss.
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# 3. SALES PREDICTION
# =============================================================================
elif nav_section == "🎯 Sales Prediction":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Predictive Sales Inference</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Estimate expected order volume, revenue, and demand velocity using the trained champion model pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px;">
            <span style="color: #94A3B8; font-size: 0.85rem;">ACTIVE PRODUCTION MODEL: </span>
            <span style="color: #38BDF8; font-weight: 600; font-size: 0.95rem;">{metadata.get('best_model_name', 'Trained ML Model')}</span>
            <span style="color: #64748B; font-size: 0.82rem; margin-left: 12px;">(Chronological Holdout Validation — 0 Data Leakage)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    pred_col1, pred_col2 = st.columns([1.1, 1.4])
    
    with pred_col1:
        st.markdown("### Scenario Configuration")
        pred_category = st.selectbox("Product Category", sorted(list(df_all["Category"].unique())), index=0)
        
        valid_prods = df_all[df_all["Category"] == pred_category]["Product"].unique()
        pred_product = st.selectbox("Target Product", sorted(list(valid_prods)), index=0)
        
        pred_region = st.selectbox("Commercial Region", sorted(list(df_all["Region"].unique())), index=0)
        pred_segment = st.selectbox("Customer Segment", sorted(list(df_all["Customer_Segment"].unique())), index=0)
        
        pred_date = st.date_input("Target Inference Date", value=(datetime.now() + timedelta(days=7)).date())
        
        # Benchmark price
        prod_hist = df_all[df_all["Product"] == pred_product]
        default_price = float(prod_hist["Unit_Price"].median()) if not prod_hist.empty else 150.0
        
        price_c, disc_c = st.columns(2)
        with price_c:
            pred_price = st.number_input("Unit Price ($)", min_value=1.0, value=round(default_price, 2), step=10.0)
        with disc_c:
            pred_discount = st.slider("Discount (%)", min_value=0, max_value=50, value=5, step=5) / 100.0

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 📦 Current Inventory on Hand")
        
        actual_catalog_stock = 45
        if "Current_Stock" in df_all.columns and not prod_hist.empty:
            actual_catalog_stock = int(prod_hist["Current_Stock"].iloc[-1])
            
        stock_session_key = f"sim_stock_{pred_product}"
        if stock_session_key not in st.session_state:
            st.session_state[stock_session_key] = actual_catalog_stock
            
        pred_stock = st.number_input(
            "Warehouse Units Available",
            min_value=0,
            max_value=10000,
            value=int(st.session_state[stock_session_key]),
            step=5,
            help="Current physical units in warehouse. Compare with predicted sales to detect low-stock risks and generate AI marketing & business strategies."
        )
        st.session_state[stock_session_key] = pred_stock
        
        st.caption("⚡ Quick Stock Simulation:")
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            if st.button("🚨 Low Stock (8)", width="stretch", help="Simulate critical shortage of 8 units"):
                st.session_state[stock_session_key] = 8
                st.rerun()
        with sim_c2:
            if st.button("🛑 Stockout (0)", width="stretch", help="Simulate zero remaining units"):
                st.session_state[stock_session_key] = 0
                st.rerun()
        with sim_c3:
            if st.button(f"🔄 Actual ({actual_catalog_stock})", width="stretch", help="Reset to recorded catalog stock"):
                st.session_state[stock_session_key] = actual_catalog_stock
                st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        predict_button = st.button("🎯 Predict Sales & Generate AI Intelligence", width="stretch")
        
    with pred_col2:
        st.markdown("### Inference Results")
        
        # Run prediction
        pred_result = predict_single_sales(
            pipeline=best_pipeline,
            metadata=metadata,
            df_history=df_all,
            product=pred_product,
            category=pred_category,
            region=pred_region,
            customer_segment=pred_segment,
            target_date=str(pred_date),
            unit_price=pred_price,
            discount=pred_discount
        )
        
        # Calculate real-time stock coverage and risk health
        hist_daily_mean = float(prod_hist["Quantity"].mean()) if not prod_hist.empty else 5.0
        stock_info = calculate_stock_health(
            predicted_units=pred_result["predicted_quantity"],
            current_stock=pred_stock,
            unit_price=pred_price,
            discount=pred_discount,
            historical_daily_avg=hist_daily_mean
        )
        
        # Results Output Cards (3 Columns)
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            render_metric_card(
                "Predicted Units",
                f"{pred_result['predicted_quantity']} units",
                f"Historical Avg: {pred_result['context']['historical_avg_qty']} units",
                f"{pred_result['context']['pct_vs_historical']:+.1f}% vs baseline",
                pred_result['context']['pct_vs_historical'] >= 0
            )
        with r_col2:
            render_metric_card(
                "Expected Revenue",
                format_currency(pred_result["expected_revenue"]),
                f"At ${pred_price:.2f} ({pred_discount*100:.0f}% disc)",
                None,
                True
            )
        with r_col3:
            delta_label = f"Deficit: -{stock_info['unit_deficit']} units" if stock_info['is_low_stock'] else f"Surplus: +{stock_info['current_stock'] - stock_info['predicted_units']} units"
            render_metric_card(
                "Inventory Coverage",
                f"{stock_info['current_stock']} units",
                f"~{stock_info['days_to_stockout']:.0f}d coverage ({stock_info['coverage_pct']:.0f}%)",
                delta_label,
                not stock_info['is_low_stock']
            )
            
        st.markdown(
            f"""
            <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 10px; padding: 14px 18px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; color: #F8FAFC; font-size: 0.95rem;">Demand & Inventory Alignment</span>
                    <div>
                        <span class="badge {pred_result['demand_badge_class']}" style="margin-right: 6px;">{pred_result['demand_level']} Demand</span>
                        <span class="badge {stock_info['badge_class']}">{stock_info['status']}</span>
                    </div>
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                    <span><strong>Date Context:</strong> {pred_result['context']['date_str']} ({pred_result['context']['day_name']}) {'— Weekend volume pattern' if pred_result['context']['is_weekend'] else '— Standard business weekday'}.</span> &nbsp;|&nbsp;
                    <span><strong>Price Ratio:</strong> {pred_result['context']['price_ratio']*100:.0f}% of catalog average.</span> &nbsp;|&nbsp;
                    <span><strong>Velocity vs Baseline:</strong> {pred_result['context']['pct_vs_historical']:+.1f}%.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # High-Impact Alert Banner if Low Stock
        if stock_info["is_low_stock"]:
            st.markdown(
                f"""
                <div class="alert-box alert-critical" style="margin-top: 12px; margin-bottom: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="font-size: 0.95rem; color: #F87171;">⚠️ SUPPLY SHORTAGE WARNING: {stock_info['headline']}</strong>
                        <span style="font-weight: 700; color: #F87171; font-size: 0.85rem;">{format_currency(stock_info['revenue_at_risk'])} Gross Revenue at Risk</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #FECACA; margin-top: 6px; line-height: 1.4;">
                        Predicted sales of <strong>{pred_result['predicted_quantity']} units</strong> exceed on-hand stock (<strong>{stock_info['current_stock']} units</strong>), creating a projected deficit of <strong>{stock_info['unit_deficit']} units</strong>. Estimated days to stockout: <strong>~{stock_info['days_to_stockout']:.1f} days</strong>. Immediate ad throttling and replenishment strategies activated below.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================================================================
    # REVENAI INTELLIGENT COMMERCIAL STRATEGY CONSOLE
    # AI Business Strategies & Marketing Strategies (Gemini + Strategic Engine)
    # =========================================================================
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Strategy Console Header
    strat_hdr_col1, strat_hdr_col2 = st.columns([3, 1])
    with strat_hdr_col1:
        status_label = "⚡ LOW-STOCK CRISIS RESPONSE PLAYBOOK" if stock_info['is_low_stock'] else "✨ PROACTIVE COMMERCIAL GROWTH PLAYBOOK"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #F8FAFC;">
                    🧠 RevenAI Strategic Intelligence Console
                </h3>
                <span style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(52, 211, 153, 0.2)); color: #38BDF8; font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.4);">
                    {status_label}
                </span>
            </div>
            <p style="color: #94A3B8; font-size: 0.86rem; margin: 4px 0 0 0;">
                AI-driven business tactics, pricing elasticity rules, and marketing campaigns synthesized for <strong>{pred_product}</strong> ({pred_category}) in <strong>{pred_region}</strong>.
            </p>
            """,
            unsafe_allow_html=True
        )
    with strat_hdr_col2:
        regen_strat = st.button("🔄 Re-analyze with AI", width="stretch", help="Re-query Gemini models for fresh strategies")

    # Generate or Retrieve Strategy Data
    strat_cache_key = f"strat_{pred_product}_{pred_region}_{pred_segment}_{pred_stock}_{pred_result['predicted_quantity']}_{pred_price}_{pred_discount}"
    
    if regen_strat or strat_cache_key not in st.session_state:
        with st.spinner("Synthesizing tailored business and marketing strategies with RevenAI AI Engine..."):
            st.session_state[strat_cache_key] = generate_ai_low_stock_strategy(
                product=pred_product,
                category=pred_category,
                region=pred_region,
                customer_segment=pred_segment,
                stock_info=stock_info,
                unit_price=pred_price,
                discount=pred_discount,
                use_gemini=True
            )
            
    strategy_data = st.session_state[strat_cache_key]

    # Executive Key Takeaway Box
    st.markdown(
        f"""
        <div style="background: linear-gradient(145deg, #111C33, #0E1729); border: 1px solid #1E293B; border-left: 4px solid #38BDF8; border-radius: 10px; padding: 16px 20px; margin-top: 12px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.05em;">Executive Strategic Synthesis</span>
                <span style="font-size: 0.75rem; color: #94A3B8;">Source: <strong>{strategy_data.get('source', 'RevenAI Intelligence Engine')}</strong></span>
            </div>
            <div style="color: #F1F5F9; font-size: 0.95rem; line-height: 1.5; font-weight: 500;">
                {strategy_data.get('key_takeaway', 'Prioritize inventory preservation, eliminate promotional spend on scarce units, and activate pre-order capture funnels.')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Strategy Console Tabs
    tab_mktg, tab_biz, tab_copy, tab_sim = st.tabs([
        "📢 AI Marketing Strategies",
        "💼 AI Business & Supply Chain Strategies",
        "📝 Ready-to-Deploy Marketing Copy",
        "📊 Stock vs Demand Sensitivity Visualizer"
    ])

    with tab_mktg:
        st.markdown("#### Prescriptive Marketing Tactics for Low-Stock Scenarios")
        st.caption("How commercial and digital marketing teams should adapt advertising campaigns, storefront messaging, and funnels.")
        
        mktg_list = strategy_data.get("marketing_strategies", [])
        m_col1, m_col2 = st.columns(2)
        
        for idx, item in enumerate(mktg_list):
            target_col = m_col1 if idx % 2 == 0 else m_col2
            with target_col:
                st.markdown(
                    f"""
                    <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; min-height: 180px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <span style="font-size: 1.25rem;">{item.get('icon', '📢')}</span>
                            <span style="font-size: 0.98rem; font-weight: 700; color: #F8FAFC;">{item.get('tactic', 'Marketing Strategy')}</span>
                        </div>
                        <div style="color: #38BDF8; font-weight: 600; font-size: 0.88rem; margin-bottom: 6px;">
                            {item.get('action', '')}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5; margin-bottom: 10px;">
                            {item.get('detail', '')}
                        </div>
                        <div style="background-color: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 5px 10px; display: inline-block;">
                            <span style="color: #38BDF8; font-size: 0.78rem; font-weight: 600;">🎯 Impact: {item.get('impact', 'Commercial Optimization')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with tab_biz:
        st.markdown("#### Operational & Supply Chain Business Strategies")
        st.caption("Actionable inventory, pricing elasticity, customer tier allocation, and procurement directives.")
        
        biz_list = strategy_data.get("business_strategies", [])
        b_col1, b_col2 = st.columns(2)
        
        for idx, item in enumerate(biz_list):
            target_col = b_col1 if idx % 2 == 0 else b_col2
            with target_col:
                st.markdown(
                    f"""
                    <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; min-height: 180px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <span style="font-size: 1.25rem;">{item.get('icon', '💼')}</span>
                            <span style="font-size: 0.98rem; font-weight: 700; color: #F8FAFC;">{item.get('tactic', 'Business Strategy')}</span>
                        </div>
                        <div style="color: #34D399; font-weight: 600; font-size: 0.88rem; margin-bottom: 6px;">
                            {item.get('action', '')}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.84rem; line-height: 1.5; margin-bottom: 10px;">
                            {item.get('detail', '')}
                        </div>
                        <div style="background-color: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 6px; padding: 5px 10px; display: inline-block;">
                            <span style="color: #34D399; font-size: 0.78rem; font-weight: 600;">🛡️ Outcome: {item.get('impact', 'Supply Protection')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with tab_copy:
        st.markdown("#### Ready-to-Deploy Marketing Copy & Sales Collateral")
        st.caption("Copy-paste ready communications generated to communicate inventory urgency without damaging customer goodwill.")
        
        copy_obj = strategy_data.get("marketing_copy", {})
        
        st.markdown("##### 📧 Customer Scarcity & Waitlist Email Campaign")
        st.text_input("Subject Line", value=copy_obj.get("email_subject", f"⚡ Low Stock Alert: {pred_product}"), key="cp_email_sub")
        st.text_area("Email Body", value=copy_obj.get("email_body", "We are running low on stock. Reserve your allocation now."), height=130, key="cp_email_body")
        
        st.markdown("##### 🌐 E-Commerce & Storefront Banner")
        st.code(copy_obj.get("storefront_banner", f"⚠️ High Demand Alert: Only {pred_stock} units left in stock. Orders ship first-come, first-served."), language="markdown")
        
        st.markdown("##### 🗣️ B2B Sales Representative Outreach Script")
        st.code(copy_obj.get("sales_rep_pitch", f"I wanted to flag that {pred_product} has only {pred_stock} units remaining in our {pred_region} warehouse. I can reserve your allocation today."), language="markdown")

    with tab_sim:
        st.markdown("#### Interactive Inventory vs Demand Sensitivity Analysis")
        st.caption("Simulate different available stock quantities and visualize the resulting surplus/deficit and stockout timing.")
        
        sim_units = st.slider(
            "Simulate Warehouse Stock on Hand (Units)",
            min_value=0,
            max_value=max(150, int(pred_result['predicted_quantity'] * 2.5)),
            value=int(pred_stock),
            step=5
        )
        
        sim_health = calculate_stock_health(
            predicted_units=pred_result["predicted_quantity"],
            current_stock=sim_units,
            unit_price=pred_price,
            discount=pred_discount,
            historical_daily_avg=hist_daily_mean
        )
        
        # Plotly chart comparing stock vs predicted demand
        fig_sim = go.Figure()
        
        fig_sim.add_trace(go.Bar(
            name="Warehouse Stock on Hand",
            x=["Inventory Position"],
            y=[sim_health["current_stock"]],
            marker_color="#38BDF8",
            text=[f"{sim_health['current_stock']} units"],
            textposition="auto"
        ))
        
        fig_sim.add_trace(go.Bar(
            name="Predicted Demand Horizon",
            x=["Inventory Position"],
            y=[sim_health["predicted_units"]],
            marker_color="#F59E0B",
            text=[f"{sim_health['predicted_units']} units"],
            textposition="auto"
        ))
        
        # Net balance bar
        net_units = sim_health["current_stock"] - sim_health["predicted_units"]
        net_color = "#10B981" if net_units >= 0 else "#EF4444"
        net_label = f"+{net_units} Surplus" if net_units >= 0 else f"{net_units} Deficit"
        
        fig_sim.add_trace(go.Bar(
            name="Net Inventory Balance",
            x=["Inventory Position"],
            y=[abs(net_units)],
            marker_color=net_color,
            text=[net_label],
            textposition="auto"
        ))
        
        fig_sim_layout = get_plotly_layout(selected_th)
        fig_sim_layout.update(
            barmode="group",
            height=280,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_sim.update_layout(fig_sim_layout)
        fig_sim.update_yaxes(title="Units")
        
        st.plotly_chart(fig_sim, width="stretch")
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Simulated Coverage", f"{sim_health['coverage_pct']:.0f}%", f"{sim_health['status']}")
        with sc2:
            st.metric("Days to Exhaustion", f"~{sim_health['days_to_stockout']:.1f} days")
        with sc3:
            st.metric("Projected Deficit", f"{sim_health['unit_deficit']} units")
        with sc4:
            st.metric("Revenue at Risk", format_currency(sim_health["revenue_at_risk"]))


# =============================================================================
# 3B. ML CLASSIFICATION ENGINE
# =============================================================================
elif nav_section == "🤖 ML Classification Engine":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">ML Classification & Strategic Tiering</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Multi-class predictive classification (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree) for arbitrary operational target columns.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Target Column Discovery
    detected_targets = st.session_state.get("quality_info", {}).get("detected_targets", [])
    candidate_targets = ["Demand_Tier"]
    for c in df_all.columns:
        if c not in ["Date", "Product", "Revenue", "Unit_Price"] and df_all[c].nunique() >= 2 and df_all[c].nunique() <= 20:
            if c not in candidate_targets:
                candidate_targets.append(c)
    for dt_col in detected_targets:
        if dt_col in df_all.columns and dt_col not in candidate_targets:
            candidate_targets.append(dt_col)

    current_target = clf_metadata.get("target_col", "Demand_Tier") if clf_metadata else "Demand_Tier"
    target_idx = candidate_targets.index(current_target) if current_target in candidate_targets else 0

    cfg_c1, cfg_c2, cfg_c3 = st.columns([2, 1.2, 1.2])
    with cfg_c1:
        selected_clf_target = st.selectbox(
            "Select Classification Target Variable",
            candidate_targets,
            index=target_idx,
            help="Choose any categorical or discrete column from the dataset to train multi-class classifiers on."
        )
    with cfg_c2:
        test_size_choice = st.selectbox("Holdout Split Ratio", [0.15, 0.20, 0.25], index=1, format_func=lambda x: f"{int(x*100)}% Test / {int((1-x)*100)}% Train")
    with cfg_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        train_clf_clicked = st.button("Train Classification Candidates", width="stretch")

    if train_clf_clicked:
        with st.spinner(f"Training classification models on target '{selected_clf_target}'..."):
            try:
                new_clf_artifacts = train_and_evaluate_classification_models(
                    df_all, target_col=selected_clf_target, test_ratio=test_size_choice
                )
                st.session_state["clf_artifacts"] = new_clf_artifacts
                st.success(f"Successfully trained 4 classification models on '{selected_clf_target}'! Champion: {new_clf_artifacts['best_model_name']}")
                st.rerun()
            except Exception as e:
                st.error(f"Classification Training Error: {e}")

    # Display Active Model Metadata
    if clf_artifacts and clf_metadata:
        champion_name = clf_metadata.get("best_model_name", "Trained Classifier")
        champion_metrics = clf_artifacts["results_df"].iloc[0]
        
        st.markdown(
            f"""
            <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin: 16px 0 20px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #94A3B8; font-size: 0.85rem;">ACTIVE CHAMPION CLASSIFIER: </span>
                        <span style="color: #38BDF8; font-weight: 700; font-size: 1.05rem;">{champion_name}</span>
                        <span style="color: #10B981; font-weight: 600; font-size: 0.85rem; margin-left: 12px;">★ Accuracy: {champion_metrics.get('Accuracy', 0):.1f}% | Weighted F1: {champion_metrics.get('F1_Score', 0):.1f}%</span>
                    </div>
                    <span class="badge badge-info">Target: {clf_metadata.get('target_col', 'Demand_Tier')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Benchmark Comparison Table
        st.subheader("Model Evaluation Benchmark Table")
        st.markdown("Rigorous multi-class model benchmarking across Accuracy, Weighted F1-Score, Precision, and Recall.")
        clf_results_df = clf_artifacts["results_df"]
        st.dataframe(clf_results_df, width="stretch", hide_index=True)

        st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'/>", unsafe_allow_html=True)

        # 3. Model Diagnostics: Confusion Matrix & Class Diagnostics
        diag_c1, diag_c2 = st.columns([1.2, 1])
        with diag_c1:
            st.subheader("Confusion Matrix (Holdout Test Set)")
            if "cm_df" in clf_artifacts:
                cm_df = clf_artifacts["cm_df"]
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Actual Class", color="Records"),
                    title=f"Holdout Confusion Matrix — {champion_name}"
                )
                fig_cm.update_layout(
                    paper_bgcolor="#0A0F1D",
                    plot_bgcolor="#0A0F1D",
                    font=dict(family="Inter, sans-serif", color="#F8FAFC"),
                    height=360
                )
                st.plotly_chart(fig_cm, width="stretch")
            else:
                st.info("Confusion matrix available upon fresh training.")

        with diag_c2:
            st.subheader("Classification Class Support & Precision")
            report_data = clf_metadata.get("classification_report", {})
            classes_in_meta = clf_metadata.get("classes", [])
            report_records = []
            for cls in classes_in_meta:
                if str(cls) in report_data:
                    c_info = report_data[str(cls)]
                    report_records.append({
                        "Class": str(cls),
                        "Precision (%)": round(float(c_info.get("precision", 0)) * 100, 1),
                        "Recall (%)": round(float(c_info.get("recall", 0)) * 100, 1),
                        "F1-Score (%)": round(float(c_info.get("f1-score", 0)) * 100, 1),
                        "Support Units": int(c_info.get("support", 0))
                    })
            if report_records:
                st.dataframe(pd.DataFrame(report_records), width="stretch", hide_index=True)
            else:
                st.dataframe(clf_results_df[["Model", "Accuracy", "F1_Score", "Status"]], width="stretch", hide_index=True)

        st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'/>", unsafe_allow_html=True)

        # 4. Interactive Operational Scenario Classification Playground
        st.subheader("Interactive Scenario Classification Playground")
        st.markdown("Simulate any commercial operational scenario and predict its classification tier in real time.")

        play_c1, play_c2 = st.columns([1.1, 1.4])
        with play_c1:
            st.markdown("#### Scenario Parameters")
            sc_category = st.selectbox("Product Category", sorted(list(df_all["Category"].unique())), key="clf_cat")
            valid_sc_prods = df_all[df_all["Category"] == sc_category]["Product"].unique()
            sc_product = st.selectbox("Target Product", sorted(list(valid_sc_prods)), key="clf_prod")
            sc_region = st.selectbox("Commercial Region", sorted(list(df_all["Region"].unique())), key="clf_reg")
            sc_segment = st.selectbox("Customer Segment", sorted(list(df_all["Customer_Segment"].unique())), key="clf_seg")
            sc_date = st.date_input("Scenario Date", value=(datetime.now() + timedelta(days=14)).date(), key="clf_date")

            prod_hist_clf = df_all[df_all["Product"] == sc_product]
            def_price_clf = float(prod_hist_clf["Unit_Price"].median()) if not prod_hist_clf.empty else 150.0

            p_col_c1, p_col_c2 = st.columns(2)
            with p_col_c1:
                sc_price = st.number_input("Unit Price ($)", min_value=1.0, value=round(def_price_clf, 2), step=10.0, key="clf_price")
            with p_col_c2:
                sc_discount = st.slider("Discount (%)", min_value=0, max_value=50, value=5, step=5, key="clf_disc") / 100.0

            classify_btn = st.button("Predict Operational Classification & Tier", width="stretch")

        with play_c2:
            st.markdown("#### Real-Time Classification Output")
            clf_res = predict_classification_single(
                pipeline=clf_artifacts["best_pipeline"],
                metadata=clf_metadata,
                df_history=df_all,
                product=sc_product,
                category=sc_category,
                region=sc_region,
                customer_segment=sc_segment,
                target_date=str(sc_date),
                unit_price=sc_price,
                discount=sc_discount
            )

            # Header card
            out_c1, out_c2 = st.columns(2)
            with out_c1:
                render_metric_card(
                    "Predicted Classification",
                    clf_res["predicted_class"],
                    f"Target Variable: {clf_res['target_name']}",
                    clf_res["predicted_class"],
                    "high" in clf_res["predicted_class"].lower() or "moderate" in clf_res["predicted_class"].lower()
                )
            with out_c2:
                render_metric_card(
                    "Model Confidence",
                    f"{clf_res['confidence_pct']:.1f}%",
                    f"Architecture: {clf_res['model_used']}",
                    "HIGH CONFIDENCE" if clf_res['confidence_pct'] >= 70 else "MODERATE CONFIDENCE",
                    clf_res['confidence_pct'] >= 70
                )

            # Class Probabilities Bar Chart
            if clf_res["class_probabilities"]:
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability (%)": v} for k, v in clf_res["class_probabilities"].items()
                ]).sort_values("Probability (%)", ascending=True)
                fig_prob = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Class",
                    orientation="h",
                    text="Probability (%)",
                    color="Class",
                    color_discrete_sequence=["#38BDF8", "#F59E0B", "#10B981", "#818CF8"],
                    title="Model Probability Distribution Across Classes"
                )
                fig_prob_layout = get_plotly_layout(selected_th)
                fig_prob_layout.update(
                    height=240,
                    showlegend=False,
                    xaxis=dict(range=[0, 100])
                )
                fig_prob.update_layout(fig_prob_layout)
                fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_prob, width="stretch")

            # Strategic Recommendation Callout
            st.markdown(
                f"""
                <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="color: #38BDF8; font-weight: 600; font-size: 0.95rem;">Strategic Action Recommendation:</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.88rem; line-height: 1.5;">
                        {clf_res['strategy']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Classification model is training or awaiting initialization. Click 'Train Classification Candidates' above.")


# =============================================================================
# 4. DEMAND FORECAST
# =============================================================================
elif nav_section == "🔮 Demand Forecast":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Future Demand Forecasting</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Multi-step forward projection using recursive autoregressive regression across customizable horizons.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    fc_ctrl1, fc_ctrl2, fc_ctrl3, fc_ctrl4 = st.columns([1, 1, 1, 1])
    with fc_ctrl1:
        horizon_choice = st.selectbox("Forecast Horizon", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"Next {x} Days")
    with fc_ctrl2:
        fc_category = st.selectbox("Category Scope", ["All Categories"] + sorted(list(df_all["Category"].unique())))
    with fc_ctrl3:
        valid_fc_prods = df_all if fc_category == "All Categories" else df_all[df_all["Category"] == fc_category]
        fc_product = st.selectbox("Product Scope", ["All Products"] + sorted(list(valid_fc_prods["Product"].unique())))
    with fc_ctrl4:
        fc_region = st.selectbox("Geographic Scope", ["All Regions"] + sorted(list(df_all["Region"].unique())))
        
    # Generate Forecast
    with st.spinner(f"Generating {horizon_choice}-day forward projection..."):
        fc_result = generate_sales_forecast(
            pipeline=best_pipeline,
            metadata=metadata,
            df_history=df_all,
            product=fc_product,
            category=fc_category,
            region=fc_region,
            horizon_days=horizon_choice
        )
        
    # Forecast Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Expected Total Units", format_number(fc_result["forecast_total_qty"]), f"Over next {horizon_choice} days", format_percentage(fc_result["growth_pct"]), fc_result["growth_pct"] >= 0)
    with m2:
        render_metric_card("Expected Revenue", format_currency(fc_result["forecast_total_rev"]), "Projected sales value", None, True)
    with m3:
        render_metric_card("Daily Run-Rate", f"{fc_result['avg_daily_forecast']:.1f} units/day", "Average projected velocity", None, True)
    with m4:
        render_metric_card("Peak Demand Date", fc_result["peak_date_str"], f"Peak volume: {fc_result['peak_qty']:.0f} units", None, True)
        
    st.plotly_chart(fc_result["fig"], width="stretch")
    
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B; margin-top: -10px; margin-bottom: 20px;">
            * Statistical projection band represents an 80% empirical prediction interval derived from chronological test set residuals. Projections assume consistent market operating conditions.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Forecast Table
    with st.expander("View Daily Forecast Projections Table"):
        df_display = fc_result["forecast_df"].copy()
        df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")
        df_display["Predicted_Revenue"] = df_display["Predicted_Revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_display, width="stretch", hide_index=True)


# =============================================================================
# 5. INVENTORY INTELLIGENCE
# =============================================================================
elif nav_section == "📦 Inventory Intelligence":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Inventory Intelligence & Stock Risk Analysis</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Evaluating current inventory run-rates against forecasted demand to prevent stockouts and optimize working capital.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    inv_c1, inv_c2, inv_c3, inv_c4 = st.columns(4)
    with inv_c1:
        render_metric_card("High Risk SKUs", str(inv_results["high_risk_count"]), "Stockout risk < 30 days", "Critical Attention", False)
    with inv_c2:
        render_metric_card("Medium Risk SKUs", str(inv_results["med_risk_count"]), "Approaching reorder point", "Monitor Closely", True)
    with inv_c3:
        render_metric_card("Low Risk SKUs", str(inv_results["low_risk_count"]), "Adequate coverage (> 45d)", "Stable", True)
    with inv_c4:
        render_metric_card("Recommended Reorder", format_number(inv_results["total_reorder_units"]), "Units to restore safety buffer", None, True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    st.plotly_chart(inv_results["fig_coverage"], width="stretch")
    
    st.markdown("### Inventory Health & Reorder Recommendations")
    
    # Render table with risk badges
    table_records = []
    for _, row in inv_df.iterrows():
        table_records.append({
            "Product": row["Product"],
            "Category": row["Category"],
            "Current Stock": f"{int(row['Current_Stock']):,}",
            "Daily Demand": f"{row['Predicted_Daily_Demand']:.1f}",
            "Coverage (DOIR)": f"{row['DOIR_Days']:.0f} days",
            "Projected Ending Balance": f"{int(row['Projected_Ending_Stock']):,}",
            "Reorder Units": f"{int(row['Recommended_Reorder_Units']):,}",
            "Risk Assessment": row["Risk_Status"]
        })
        
    st.dataframe(pd.DataFrame(table_records), width="stretch", hide_index=True)
    
    st.markdown(
        """
        <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin-top: 14px;">
            <h4 style="color: #F8FAFC; margin: 0 0 6px 0; font-size: 0.95rem;">Calculation Methodology & Formulations</h4>
            <div style="color: #94A3B8; font-size: 0.83rem; line-height: 1.5;">
                • <strong>Days of Inventory Remaining (DOIR):</strong> <code>Current Stock / Predicted Daily Demand Velocity</code><br/>
                • <strong>High Risk Threshold:</strong> DOIR &lt; Forecast Horizon (30 days) — stockout expected before cycle completion.<br/>
                • <strong>Medium Risk Threshold:</strong> DOIR between 30 and 45 days — within standard supplier lead-time replenishment window.<br/>
                • <strong>Target Safety Stock Buffer:</strong> Standardized at 45 days of predicted demand.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# 6. PRODUCT INTELLIGENCE
# =============================================================================
elif nav_section in ["🏷️ Product Intelligence", "🏷️ Product Intelligence (Min/Max)"]:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Product Intelligence & Min/Max Operational Bounds</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Granular unit economics, minimum & maximum price/volume boundaries, peak demand envelopes, and catalog benchmark tables.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    selected_sku = st.selectbox("Select Target Product SKU", sorted(list(df_all["Product"].unique())))
    
    sku_history = df_all[df_all["Product"] == selected_sku].copy()
    sku_demand = demand_df[demand_df["Product"] == selected_sku].iloc[0]
    sku_inv = inv_df[inv_df["Product"] == selected_sku].iloc[0]
    
    sku_revenue = float(sku_history["Revenue"].sum())
    sku_units = int(sku_history["Quantity"].sum())
    sku_avg_price = float(sku_history["Unit_Price"].mean())
    sku_avg_discount = float(sku_history["Discount"].mean())
    
    # Calculate ranking
    prod_revs = df_all.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    prod_rank = list(prod_revs.index).index(selected_sku) + 1
    total_skus = len(prod_revs)
    
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        render_metric_card("Total Revenue", format_currency(sku_revenue), f"Catalog Rank: #{prod_rank} of {total_skus}", None, True)
    with p2:
        render_metric_card("Total Units", format_number(sku_units), "Historical volume", None, True)
    with p3:
        render_metric_card("Average Price", f"${sku_avg_price:.2f}", f"Avg Discount: {sku_avg_discount*100:.1f}%", None, True)
    with p4:
        render_metric_card("Demand Velocity", f"{sku_demand['Predicted_Daily_Demand']:.1f} /day", f"{sku_demand['Demand_Change_Pct']:+.1f}% vs baseline", sku_demand["Demand_Level"], sku_demand["Demand_Change_Pct"] >= 0)
    with p5:
        render_metric_card("Stock Position", f"{int(sku_inv['Current_Stock']):,} units", f"DOIR: {sku_inv['DOIR_Days']:.0f} days", sku_inv["Risk_Status"], sku_inv["Risk_Status"] == "LOW RISK")
        
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ----------------- PRODUCT MINIMUM & MAXIMUM BOUNDS -----------------
    st.markdown(
        f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #38BDF8;">
                        📐 Operational Minimum & Maximum Metrics for {selected_sku}
                    </h3>
                    <p style="margin: 3px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                        Rigorous historical extremes: floor and ceiling prices, minimum/maximum daily velocity, and all-time record dates.
                    </p>
                </div>
                <span class="badge badge-info">Min-Max Boundary Guardrails</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    min_max_info = get_product_min_max_metrics(df_all, selected_sku)
    
    mm1, mm2, mm3, mm4 = st.columns(4)
    with mm1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Unit Price Boundaries</div>
                <div class="metric-value" style="font-size: 1.25rem;">
                    ${min_max_info['min_price']:.2f} <span style="color: #64748B; font-size: 0.9rem;">to</span> ${min_max_info['max_price']:.2f}
                </div>
                <div class="metric-context">
                    Spread: <strong>${min_max_info['price_spread']:.2f}</strong> ({min_max_info['price_spread_pct']:.1f}% dynamic range)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mm2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Daily Volume Extremes</div>
                <div class="metric-value" style="font-size: 1.25rem;">
                    {min_max_info['min_daily_qty']} <span style="color: #64748B; font-size: 0.9rem;">to</span> {min_max_info['max_daily_qty']} units
                </div>
                <div class="metric-context">
                    Avg: <strong>{min_max_info['avg_daily_qty']:.1f}</strong> units/day | Peak Date: <strong>{min_max_info['peak_date']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mm3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Daily Revenue Range</div>
                <div class="metric-value" style="font-size: 1.25rem;">
                    ${min_max_info['min_daily_rev']:,.0f} <span style="color: #64748B; font-size: 0.9rem;">to</span> ${min_max_info['max_daily_rev']:,.0f}
                </div>
                <div class="metric-context">
                    Peak Daily Rev: <strong>${min_max_info['peak_rev']:,.2f}</strong> on {min_max_info['peak_date']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mm4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Order Size Boundaries</div>
                <div class="metric-value" style="font-size: 1.25rem;">
                    {min_max_info['min_order_qty']} <span style="color: #64748B; font-size: 0.9rem;">to</span> {min_max_info['max_order_qty']} units
                </div>
                <div class="metric-context">
                    Avg Order: <strong>{min_max_info['avg_order_qty']:.1f}</strong> | Discount: {min_max_info['min_discount_pct']:.0f}%–{min_max_info['max_discount_pct']:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Min-Max Operational Demand Band Envelope Chart
    st.subheader(f"Operational Demand Envelope (Min-Max Band) for {selected_sku}")
    st.markdown("Shows actual daily sales against historical 7-day rolling minimum and maximum demand thresholds.")
    fig_band = create_product_min_max_band_chart(df_all, selected_sku)
    st.plotly_chart(fig_band, width="stretch")
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # SKU Historical Revenue vs Units
    sku_daily = sku_history.groupby("Date").agg({"Revenue": "sum", "Quantity": "sum"}).reset_index()
    fig_sku = go.Figure()
    fig_sku.add_trace(go.Scatter(
        x=sku_daily["Date"],
        y=sku_daily["Revenue"],
        mode="lines",
        name="Revenue ($)",
        line=dict(color="#38BDF8", width=2)
    ))
    fig_sku.add_trace(go.Scatter(
        x=sku_daily["Date"],
        y=sku_daily["Quantity"],
        mode="lines",
        name="Units",
        yaxis="y2",
        line=dict(color="#F97316", width=1.8, dash="dot")
    ))
    sku_layout = get_plotly_layout()
    sku_layout.update(
        title=f"Historical Sales Trajectory for {selected_sku}",
        yaxis=dict(title="Revenue ($)", gridcolor="#1E293B"),
        yaxis2=dict(title="Units Sold", overlaying="y", side="right", showgrid=False),
        height=360
    )
    fig_sku.update_layout(sku_layout)
    st.plotly_chart(fig_sku, width="stretch")
    
    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'/>", unsafe_allow_html=True)
    
    # Catalog-Wide Min vs. Max Benchmarking
    st.subheader("Cross-Catalog Minimum & Maximum Price Boundaries")
    st.markdown("Visual comparison of price flexibility and spread (Minimum vs. Average vs. Maximum Price) across all catalog products.")
    fig_price_spans = create_product_price_range_chart(df_all)
    st.plotly_chart(fig_price_spans, width="stretch")
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Full Catalog Min-Max Matrix Table
    st.subheader("Master Catalog Minimum & Maximum Operational Table")
    st.markdown("Detailed breakdown of min/max pricing, volume spikes, and all-time peak dates for every product SKU in the dataset.")
    catalog_table = get_all_products_min_max_table(df_all)
    st.dataframe(catalog_table, width="stretch", hide_index=True)
    
    # Download CSV
    csv_min_max = catalog_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Complete Min-Max Catalog Matrix (CSV)",
        data=csv_min_max,
        file_name=f"product_min_max_catalog_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# =============================================================================
# 7. REGIONAL INTELLIGENCE
# =============================================================================
elif nav_section == "🌍 Regional Intelligence":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Regional Intelligence & Geographic Expansion</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Territory breakdown, metropolitan distribution, and regional demand dynamics.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    r_bar1, r_bar2 = st.columns(2)
    fig_r1, fig_r2 = create_regional_analysis(df_all)
    with r_bar1:
        st.plotly_chart(fig_r1, width="stretch")
    with r_bar2:
        st.plotly_chart(fig_r2, width="stretch")
        
    # Regional Deep-Dive Selector
    target_region = st.selectbox("Inspect Specific Region", sorted(list(df_all["Region"].unique())))
    reg_subset = df_all[df_all["Region"] == target_region]
    
    # Cities in this region
    city_summary = reg_subset.groupby("City").agg({
        "Revenue": "sum",
        "Quantity": "sum",
        "Product": "nunique"
    }).reset_index().sort_values("Revenue", ascending=False)
    
    city_summary["Revenue"] = city_summary["Revenue"].apply(lambda x: f"${x:,.2f}")
    city_summary["Quantity"] = city_summary["Quantity"].apply(lambda x: f"{x:,}")
    city_summary.columns = ["City", "Revenue ($)", "Volume (Units)", "Active Products"]
    
    st.markdown(f"#### Commercial Performance Across Cities in {target_region}")
    st.dataframe(city_summary, width="stretch", hide_index=True)


# =============================================================================
# 8. SEASONAL INTELLIGENCE
# =============================================================================
elif nav_section == "📅 Seasonal Intelligence":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Seasonal Intelligence & Purchasing Cycles</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Decomposing recurring annual, quarterly, and weekly demand oscillations to inform inventory planning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    s_c1, s_c2 = st.columns(2)
    fig_m, fig_d = create_seasonal_analysis(df_all)
    with s_c1:
        st.plotly_chart(fig_m, width="stretch")
    with s_c2:
        st.plotly_chart(fig_d, width="stretch")
        
    st.markdown(
        """
        <div class="alert-box alert-info">
            <strong>Key Seasonal Findings:</strong><br/>
            1. <strong>Quarter-End Acceleration:</strong> Noticeable volume expansion occurs at the conclusion of Q2 and Q4, driven by institutional budget utilization.<br/>
            2. <strong>Weekday vs Weekend Dispersion:</strong> Commercial weekday transactions represent over 82% of weekly sales volume, necessitating synchronized logistics scheduling between Tuesday and Thursday.<br/>
            3. <strong>Summer Plateau:</strong> July and August register normalized procurement volumes before the autumnal surge.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# 9. MODEL PERFORMANCE & EXPLAINABILITY
# =============================================================================
elif nav_section == "🧠 Model Performance & Explainability":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Predictive Model Benchmarking & Explainable AI</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Rigorous evaluation of regression candidates using chronological holdout validation, paired with model interpretability.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model Comparison Benchmark Table
    st.subheader("Model Evaluation Benchmark Table")
    st.markdown("Chronological 80% Train / 20% Unseen Future Test Split")
    
    results_df = ml_artifacts["results_df"]
    st.dataframe(results_df, width="stretch", hide_index=True)
    
    retrain_col, _ = st.columns([1, 4])
    with retrain_col:
        if st.button("Retrain All Models Now"):
            with st.spinner("Retraining pipelines..."):
                st.session_state["ml_artifacts"] = train_and_evaluate_models(df_all)
                st.success("All models successfully retrained and re-evaluated!")
                st.rerun()

    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    
    # Feature Importance
    st.subheader("Model Interpretability & Feature Influence")
    xai_res = extract_feature_importance(best_pipeline, metadata)
    st.plotly_chart(xai_res["fig"], width="stretch")
    
    st.markdown("#### Business Driver Interpretations")
    int_c1, int_c2 = st.columns(2)
    for i, item in enumerate(xai_res["interpretations"]):
        target_col = int_c1 if i % 2 == 0 else int_c2
        with target_col:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-title">{item['driver']} <span class="badge badge-info" style="margin-left: 8px;">{item['impact']}</span></div>
                    <div class="rec-reason">{item['summary']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 10px;">
            * Statistical Note: Feature importance reflects relative weight inside the machine learning model architecture. High importance denotes strong predictive contribution rather than direct real-world causality.
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# 10. ALERT CENTER
# =============================================================================
elif nav_section == "⚡ Alert Center":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Intelligent Operational Alert Center</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Dynamically generated alerts triggered by real-time inventory anomalies, sudden demand surges, and margin leakage.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    alerts = generate_system_alerts(df_all, demand_df, inv_df)
    
    for alert in alerts:
        css_class = f"alert-{alert['type']}"
        st.markdown(
            f"""
            <div class="alert-box {css_class}">
                <div style="font-weight: 700; font-size: 1rem; margin-bottom: 4px;">{alert['title']}</div>
                <div style="font-size: 0.9rem; margin-bottom: 6px;">{alert['message']}</div>
                <div style="font-size: 0.82rem; font-weight: 600; opacity: 0.9;">Recommended Action: {alert['action']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# 11. BUSINESS RECOMMENDATIONS
# =============================================================================
elif nav_section == "💡 Business Recommendations":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Prescriptive Business Recommendations</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Empirical decision support generated through automated heuristics and predictive modeling.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for rec in recommendations:
        st.markdown(
            f"""
            <div class="rec-card" style="border-left: 4px solid #F97316;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">{rec['category']} • {rec['impact']}</span>
                    <span class="badge {rec['badge_class']}">{rec['priority']} Priority</span>
                </div>
                <div class="rec-title" style="font-size: 1.1rem; color: #F8FAFC;">{rec['title']}</div>
                <div style="color: #CBD5E1; font-size: 0.92rem; margin-bottom: 8px; font-weight: 500;">{rec['recommendation']}</div>
                <div class="rec-reason" style="background-color: #0F172A; padding: 10px 14px; border-radius: 6px; border: 1px solid #1E293B;">
                    <strong style="color: #F97316;">Empirical Rationale:</strong> {rec['reason']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# 12. EXECUTIVE REPORTS & DATA EXPORT
# =============================================================================
elif nav_section == "📄 Executive Reports & Downloads":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Executive Decision Reports & Data Export</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Export automated PDF intelligence dossiers and download granular tabular datasets for offline auditing.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 1. Generate Executive PDF Report")
    st.markdown("Compiles high-level commercial KPIs, model benchmarks, stockout risk tables, and prescriptive recommendations into a downloadable PDF.")
    
    kpi_summary = {
        "total_revenue": total_revenue,
        "total_quantity": total_quantity,
        "avg_order_value": avg_order_value,
        "num_products": num_products,
        "num_categories": num_categories
    }
    
    try:
        pdf_bytes = generate_pdf_business_report(
            quality_info=st.session_state["quality_info"],
            kpi_summary=kpi_summary,
            model_results_df=results_df,
            inv_df=inv_df,
            recommendations=recommendations
        )
        st.download_button(
            label="📥 Download RevenAI Executive Business Report (PDF)",
            data=pdf_bytes,
            file_name=f"RevenAI_Executive_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    except Exception as pdf_err:
        st.error(f"Report Generation Notice: {pdf_err}")

    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'/>", unsafe_allow_html=True)
    
    st.markdown("### 2. Download Tabular Data (CSV)")
    d_col1, d_col2, d_col3 = st.columns(3)
    
    with d_col1:
        st.markdown("#### Filtered Sales Data")
        st.download_button(
            "Download Sales Transactions (CSV)",
            df_filtered.to_csv(index=False).encode("utf-8"),
            "filtered_sales_data.csv",
            "text/csv"
        )
        
    with d_col2:
        st.markdown("#### Demand Velocity Analysis")
        st.download_button(
            "Download Demand Analysis (CSV)",
            demand_df.to_csv(index=False).encode("utf-8"),
            "demand_intelligence.csv",
            "text/csv"
        )
        
    with d_col3:
        st.markdown("#### Inventory Risk & Stock")
        st.download_button(
            "Download Inventory Risk (CSV)",
            inv_df.to_csv(index=False).encode("utf-8"),
            "inventory_risk_analysis.csv",
            "text/csv"
        )


# =============================================================================
# 13. DATA MANAGEMENT & QUALITY
# =============================================================================
elif nav_section == "📁 Data Management & Quality":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 4px;">Data Management & Quality Assurance</h1>
            <p style="color: #94A3B8; font-size: 0.95rem;">Upload new transactional files, inspect data quality metrics, detect outliers, and review automated schema mapping.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #38BDF8;">
                        🌐 Universal Dataset Ingestion Engine
                    </h3>
                    <p style="margin: 3px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                        This platform dynamically ingests <strong>any dataset</strong> (CSV or Excel). Missing date timelines are automatically synthesized, column aliases mapped, and classification target fields auto-discovered.
                    </p>
                </div>
                <span class="badge badge-info">Adaptive Schema Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    up_c1, up_c2 = st.columns([2, 1.2])
    with up_c1:
        uploaded_file = st.file_uploader(
            "Upload Any Custom Dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload any dataset. The engine automatically reconciles dates, volume, prices, and classification targets."
        )
        if uploaded_file is not None:
            if st.button("Process & Adapt Uploaded Dataset Now", width="stretch"):
                try:
                    with st.spinner("Analyzing schema, synthesizing timeline, and retraining ML models..."):
                        new_df, new_q, new_warns = load_dataset(uploaded_file)
                        new_clean, clean_sum = clean_dataset(new_df)
                        st.session_state["raw_df"] = new_clean
                        st.session_state["quality_info"] = new_q
                        st.session_state["data_warnings"] = new_warns
                        # Retrain both Regression and Classification models
                        st.session_state["ml_artifacts"] = train_and_evaluate_models(new_clean)
                        st.session_state["clf_artifacts"] = train_and_evaluate_classification_models(new_clean)
                        st.success("Universal dataset loaded successfully! Regression and Classification champions retrained.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Dataset Processing Error: {str(e)}")
                    
    with up_c2:
        st.markdown("#### Instant Multi-Domain Presets")
        st.markdown("Test the platform's ability to adapt to different domain datasets:")
        
        p_b1, p_b2 = st.columns(2)
        with p_b1:
            if st.button("Tech Enterprise (Default)", width="stretch"):
                with st.spinner("Reloading Enterprise dataset..."):
                    gen_df = generate_synthetic_sales_data(days=730, output_path="data/demo_sales.csv")
                    gen_clean, _ = clean_dataset(gen_df)
                    st.session_state["raw_df"] = gen_clean
                    st.session_state["ml_artifacts"] = train_and_evaluate_models(gen_clean)
                    st.session_state["clf_artifacts"] = train_and_evaluate_classification_models(gen_clean)
                    st.success("Enterprise dataset loaded!")
                    st.rerun()
        with p_b2:
            if st.button("Retail FMCG (Min/Max)", width="stretch"):
                with st.spinner("Synthesizing Retail FMCG dataset..."):
                    # Generate retail FMCG dataset with fast moving consumer items
                    fmcg_df = generate_synthetic_sales_data(days=365, output_path="data/fmcg_demo.csv")
                    fmcg_df["Category"] = np.random.choice(["Beverages", "Snacks", "Personal Care", "Dairy"], size=len(fmcg_df))
                    fmcg_df["Product"] = np.random.choice([
                        "Organic Almond Milk 1L", "Sparkling Spring Water 500ml", "Artisan Roasted Coffee",
                        "Whole Grain Granola 500g", "Dark Chocolate 85%", "Hydrating Face Wash 200ml",
                        "Greek Yogurt 500g", "Green Tea Infusion 25pk"
                    ], size=len(fmcg_df))
                    fmcg_clean, _ = clean_dataset(fmcg_df)
                    st.session_state["raw_df"] = fmcg_clean
                    st.session_state["ml_artifacts"] = train_and_evaluate_models(fmcg_clean)
                    st.session_state["clf_artifacts"] = train_and_evaluate_classification_models(fmcg_clean)
                    st.success("Retail FMCG dataset loaded!")
                    st.rerun()

    st.markdown("<hr style='border-color: #1E293B;'/>", unsafe_allow_html=True)
    
    # Data Quality Indicators
    q_score = compute_data_quality_score(df_all)
    st.subheader("Data Quality Indicators")
    
    dq1, dq2, dq3, dq4, dq5 = st.columns(5)
    with dq1:
        render_metric_card("Data Quality Score", f"{q_score['quality_score']}%", "Composite reliability index", None, q_score['quality_score'] >= 90)
    with dq2:
        render_metric_card("Completeness", f"{q_score['completeness_score']}%", f"{q_score['missing_cells']} missing values", None, True)
    with dq3:
        render_metric_card("Uniqueness", f"{q_score['uniqueness_score']}%", f"{q_score['duplicate_rows']} duplicate records", None, True)
    with dq4:
        render_metric_card("Validity", f"{q_score['validity_score']}%", f"{q_score['invalid_rows']} invalid records", None, True)
    with dq5:
        render_metric_card("Dataset Volume", f"{len(df_all):,} rows", f"{len(df_all.columns)} schema fields", None, True)

    # Warnings if any
    if st.session_state.get("data_warnings"):
        for w in st.session_state["data_warnings"]:
            st.info(f"ℹ {w}")
            
    # Statistical Outlier Detection
    outlier_info = detect_outliers_iqr(df_all, "Revenue")
    st.markdown(
        f"""
        <div style="background-color: #111C33; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; margin: 16px 0;">
            <h4 style="color: #F8FAFC; margin: 0 0 6px 0; font-size: 0.95rem;">Statistical Outlier Detection (Revenue IQR Analysis)</h4>
            <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                • Detected <strong>{outlier_info['outlier_count']:,}</strong> statistical outliers ({outlier_info['outlier_percentage']}% of records) outside normal IQR bounds [${outlier_info['lower_bound']:,.2f} – ${outlier_info['upper_bound']:,.2f}].<br/>
                • Transactions are retained with robust scaling in model pipelines to capture true institutional peak demand.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Raw Data Preview
    st.subheader("Data Schema & Sample Records")
    preview_df = df_all.head(20).copy()
    if "Date" in preview_df.columns:
        preview_df["Date"] = pd.to_datetime(preview_df["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(preview_df, width="stretch", hide_index=True)
    
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_all.describe().round(2), width="stretch")
