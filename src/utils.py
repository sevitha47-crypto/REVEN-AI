"""
Utility functions and UI styling helpers for RevenAI — Smart Sales Prediction System.
Supports executive Dark Navy and Crisp Light themes with dynamic switching.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

def get_current_theme():
    """Retrieve active theme name ('dark' or 'light') from session state."""
    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "dark"
    return st.session_state["app_theme"]


def apply_custom_css(theme: str = None):
    """Inject polished executive CSS styles into the Streamlit app based on selected theme."""
    if theme is None:
        theme = get_current_theme()

    if theme == "light":
        # Professional executive crisp light palette
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Light Main Container */
        .stApp {
            background-color: #F8FAFC !important;
            color: #1E293B !important;
        }

        /* Light Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #F1F5F9 !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #0F172A !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p {
            color: #334155 !important;
        }

        /* Cards */
        .metric-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px;
            padding: 20px 22px;
            box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.08);
            margin-bottom: 14px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: #2563EB !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px -2px rgba(37, 99, 235, 0.12);
        }
        .metric-title {
            color: #64748B !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: #0F172A !important;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            margin-top: 6px;
            color: #64748B !important;
        }
        .metric-delta-pos {
            color: #059669 !important;
            font-weight: 600;
            font-size: 0.82rem;
        }
        .metric-delta-neg {
            color: #DC2626 !important;
            font-weight: 600;
            font-size: 0.82rem;
        }

        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-high {
            background-color: #FEE2E2 !important;
            color: #B91C1C !important;
            border: 1px solid #FCA5A5 !important;
        }
        .badge-medium {
            background-color: #FEF3C7 !important;
            color: #B45309 !important;
            border: 1px solid #FCD34D !important;
        }
        .badge-low {
            background-color: #D1FAE5 !important;
            color: #047857 !important;
            border: 1px solid #6EE7B7 !important;
        }
        .badge-info {
            background-color: #DBEAFE !important;
            color: #1D4ED8 !important;
            border: 1px solid #93C5FD !important;
        }

        /* Alert Boxes */
        .alert-box {
            padding: 16px 20px;
            border-radius: 10px;
            margin-bottom: 14px;
            border-left: 4px solid;
        }
        .alert-critical {
            background-color: #FEF2F2 !important;
            border-left-color: #DC2626 !important;
            color: #991B1B !important;
            border-top: 1px solid #FEE2E2;
            border-right: 1px solid #FEE2E2;
            border-bottom: 1px solid #FEE2E2;
        }
        .alert-warning {
            background-color: #FFFBEB !important;
            border-left-color: #D97706 !important;
            color: #92400E !important;
            border-top: 1px solid #FEF3C7;
            border-right: 1px solid #FEF3C7;
            border-bottom: 1px solid #FEF3C7;
        }
        .alert-success {
            background-color: #F0FDF4 !important;
            border-left-color: #16A34A !important;
            color: #166534 !important;
            border-top: 1px solid #DCFCE7;
            border-right: 1px solid #DCFCE7;
            border-bottom: 1px solid #DCFCE7;
        }
        .alert-info {
            background-color: #EFF6FF !important;
            border-left-color: #2563EB !important;
            color: #1E40AF !important;
            border-top: 1px solid #DBEAFE;
            border-right: 1px solid #DBEAFE;
            border-bottom: 1px solid #DBEAFE;
        }

        /* Recommendation Card */
        .rec-card {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }
        .rec-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #0F172A !important;
            margin-bottom: 4px;
        }
        .rec-reason {
            font-size: 0.85rem;
            color: #475569 !important;
            line-height: 1.45;
        }

        /* Custom Streamlit Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(180deg, #EA580C 0%, #C2410C 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.5rem 1.2rem;
            box-shadow: 0 2px 8px rgba(234, 88, 12, 0.25);
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(180deg, #F97316 0%, #EA580C 100%) !important;
            box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4);
            transform: translateY(-1px);
        }
        div.stDownloadButton > button {
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        div.stDownloadButton > button:hover {
            border-color: #2563EB !important;
            color: #2563EB !important;
        }

        /* Headers & Subheaders */
        h1, h2, h3, h4 {
            color: #0F172A !important;
            letter-spacing: -0.02em;
        }
        p, span, label {
            color: #334155;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #64748B;
            font-weight: 500;
            padding: 10px 16px;
            border-radius: 6px 6px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #EA580C !important;
            border-bottom: 2px solid #EA580C !important;
            font-weight: 600;
        }

        /* Inputs & Widgets in light mode */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        div[data-baseweb="select"] * {
            color: #0F172A !important;
        }
        .stDateInput input, .stNumberInput input, .stTextInput input {
            color: #0F172A !important;
            background-color: #FFFFFF !important;
        }

        /* Dataframe Table styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            overflow: hidden;
            background: #FFFFFF;
        }

        /* Custom content box overrides for light theme */
        div[style*="background-color: #111C33"],
        div[style*="background-color:#111C33"],
        div[style*="background-color: #0F172A"],
        div[style*="background-color:#0F172A"] {
            background-color: #FFFFFF !important;
            border-color: #E2E8F0 !important;
            color: #1E293B !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }

        div[style*="background: linear-gradient(145deg, #111C33, #0E1729)"] {
            background: linear-gradient(145deg, #FFFFFF, #F8FAFC) !important;
            border-color: #E2E8F0 !important;
            color: #1E293B !important;
        }

        /* Fix text colors inside styled cards in light mode */
        div[style*="background-color: #111C33"] span[style*="color: #F8FAFC"],
        div[style*="background-color: #111C33"] div[style*="color: #F8FAFC"],
        div[style*="background-color: #111C33"] h4[style*="color: #F8FAFC"],
        div[style*="background-color: #0F172A"] span[style*="color: #F8FAFC"],
        div[style*="background: linear-gradient(145deg, #111C33, #0E1729)"] div[style*="color: #F1F5F9"] {
            color: #0F172A !important;
        }

        div[style*="background-color: #111C33"] span[style*="color: #94A3B8"],
        div[style*="background-color: #111C33"] div[style*="color: #94A3B8"],
        div[style*="background-color: #0F172A"] p[style*="color: #94A3B8"] {
            color: #475569 !important;
        }

        hr[style*="border-color: #1E293B"] {
            border-color: #E2E8F0 !important;
        }
        </style>
        """
    else:
        # High-contrast executive dark navy palette
        custom_css = """
        <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main Container background */
        .stApp {
            background-color: #0B1120 !important;
            color: #E2E8F0 !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #070D1B !important;
            border-right: 1px solid #1E293B !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #F8FAFC !important;
        }

        /* Cards */
        .metric-card {
            background: linear-gradient(145deg, #111C33, #0F172A);
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 20px 22px;
            box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.4);
            margin-bottom: 14px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: #3B82F6;
            transform: translateY(-2px);
        }
        .metric-title {
            color: #94A3B8;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            margin-top: 6px;
            color: #64748B;
        }
        .metric-delta-pos {
            color: #10B981;
            font-weight: 600;
            font-size: 0.82rem;
        }
        .metric-delta-neg {
            color: #F43F5E;
            font-weight: 600;
            font-size: 0.82rem;
        }

        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .badge-high {
            background-color: rgba(239, 68, 68, 0.18);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .badge-medium {
            background-color: rgba(245, 158, 11, 0.18);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .badge-low {
            background-color: rgba(16, 185, 129, 0.18);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .badge-info {
            background-color: rgba(59, 130, 246, 0.18);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }

        /* Alert Boxes */
        .alert-box {
            padding: 16px 20px;
            border-radius: 10px;
            margin-bottom: 14px;
            border-left: 4px solid;
        }
        .alert-critical {
            background-color: #1F151B;
            border-left-color: #EF4444;
            color: #FECACA;
        }
        .alert-warning {
            background-color: #1F1A14;
            border-left-color: #F59E0B;
            color: #FEF3C7;
        }
        .alert-success {
            background-color: #11221B;
            border-left-color: #10B981;
            color: #D1FAE5;
        }
        .alert-info {
            background-color: #111A2E;
            border-left-color: #3B82F6;
            color: #DBEAFE;
        }

        /* Recommendation Card */
        .rec-card {
            background-color: #111C33;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .rec-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #F8FAFC;
            margin-bottom: 4px;
        }
        .rec-reason {
            font-size: 0.85rem;
            color: #94A3B8;
            line-height: 1.45;
        }

        /* Custom Streamlit Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(180deg, #F97316 0%, #EA580C 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.5rem 1.2rem;
            box-shadow: 0 2px 8px rgba(249, 115, 22, 0.35);
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(180deg, #FB923C 0%, #F97316 100%);
            box-shadow: 0 4px 14px rgba(249, 115, 22, 0.5);
            transform: translateY(-1px);
        }
        div.stDownloadButton > button {
            background: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            font-weight: 500;
        }
        div.stDownloadButton > button:hover {
            border-color: #3B82F6 !important;
            color: #60A5FA !important;
        }

        /* Headers & Subheaders */
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            letter-spacing: -0.02em;
        }
        p, span, label {
            color: #CBD5E1;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #1E293B;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #94A3B8;
            font-weight: 500;
            padding: 10px 16px;
            border-radius: 6px 6px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #F97316 !important;
            border-bottom: 2px solid #F97316 !important;
            font-weight: 600;
        }

        /* Dataframe Table styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #1E293B;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)


def get_plotly_layout(theme: str = None):
    """Return theme-adaptive (Dark or Light) executive Plotly layout parameters."""
    if theme is None:
        theme = get_current_theme()

    if theme == "light":
        return dict(
            paper_bgcolor="rgba(255, 255, 255, 0)",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            margin=dict(l=45, r=25, t=45, b=45),
            xaxis=dict(
                gridcolor="#E2E8F0",
                linecolor="#CBD5E1",
                tickcolor="#CBD5E1",
                tickfont=dict(color="#475569"),
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="#E2E8F0",
                linecolor="#CBD5E1",
                tickcolor="#CBD5E1",
                tickfont=dict(color="#475569"),
                showgrid=True,
                zeroline=False,
            ),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#CBD5E1",
                borderwidth=1,
                font=dict(color="#1E293B", size=11),
            ),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#2563EB",
                font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
            ),
        )
    else:
        return dict(
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(17, 28, 51, 0.45)",
            font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
            margin=dict(l=45, r=25, t=45, b=45),
            xaxis=dict(
                gridcolor="#1E293B",
                linecolor="#334155",
                tickcolor="#334155",
                tickfont=dict(color="#94A3B8"),
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="#1E293B",
                linecolor="#334155",
                tickcolor="#334155",
                tickfont=dict(color="#94A3B8"),
                showgrid=True,
                zeroline=False,
            ),
            legend=dict(
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="#1E293B",
                borderwidth=1,
                font=dict(color="#E2E8F0", size=11),
            ),
            hoverlabel=dict(
                bgcolor="#0F172A",
                bordercolor="#3B82F6",
                font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
            ),
        )


def format_currency(val):
    """Format numeric value as USD currency."""
    if val is None or val != val:
        return "$0.00"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"${val / 1_000:.1f}K"
    else:
        return f"${val:,.2f}"


def format_number(val, decimals=0):
    """Format standard number with thousands separator."""
    if val is None or val != val:
        return "0"
    if decimals == 0:
        return f"{int(round(val)):,}"
    return f"{val:,.{decimals}f}"


def format_percentage(val):
    """Format percentage with +/- prefix."""
    if val is None or val != val:
        return "0.0%"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}%"


def render_metric_card(title: str, value: str, subtitle: str = "", delta: str = None, is_positive: bool = True):
    """Render HTML KPI card with consistent typography."""
    delta_html = ""
    if delta:
        css_class = "metric-delta-pos" if is_positive else "metric-delta-neg"
        delta_html = f'<div class="{css_class}">{delta}</div>'

    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
