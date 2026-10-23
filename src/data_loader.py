"""
Data loading, column mapping, schema validation, and synthetic sales generation.
"""

import os
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Comprehensive alias dictionaries for automatic column detection across any domain
COLUMN_ALIASES = {
    "date": [
        "date", "order_date", "transaction_date", "sale_date", "invoice_date", "timestamp", "dt",
        "time", "day", "orderdate", "datetime", "period", "created_at", "trans_date", "event_date"
    ],
    "product": [
        "product", "product_name", "item", "item_name", "sku", "product_id", "prod_name",
        "title", "name", "article", "description", "model", "item_identifier", "service", "course"
    ],
    "category": [
        "category", "product_category", "dept", "department", "item_category", "segment_category",
        "type", "genre", "class", "group", "sector", "division", "item_type", "sub_category"
    ],
    "quantity": [
        "quantity", "qty", "units", "volume", "order_quantity", "units_sold", "count",
        "amount_sold", "pieces", "items", "size", "weight", "num_items", "sales_count"
    ],
    "unit_price": [
        "unit_price", "price", "rate", "item_price", "selling_price", "unitprice", "cost",
        "mrp", "list_price", "value", "fare", "fee", "charge", "item_mrp"
    ],
    "discount": [
        "discount", "discount_rate", "disc", "discount_pct", "discount_percent", "promo_discount",
        "rebate", "reduction", "coupon", "discount_amount"
    ],
    "revenue": [
        "revenue", "sales", "total_sales", "total_revenue", "amount", "total_amount", "net_sales",
        "order_value", "total", "turnover", "gross_sales", "gross_revenue", "income", "item_outlet_sales", "price_total"
    ],
    "region": [
        "region", "territory", "zone", "market", "area", "geography", "country", "state",
        "province", "location", "store", "branch", "continent", "outlet_location_type"
    ],
    "city": ["city", "location", "town", "branch", "metro_area", "district", "outlet_identifier"],
    "customer_segment": [
        "customer_segment", "segment", "customer_type", "client_segment", "tier",
        "account_type", "customer_group", "user_type", "plan", "loyalty_tier"
    ],
    "current_stock": [
        "current_stock", "stock", "inventory", "inventory_level", "stock_on_hand",
        "units_in_stock", "stock_quantity", "available_stock"
    ],
    "target_label": [
        "target", "status", "class", "label", "outcome", "churn", "exited", "decision",
        "rating", "sentiment", "risk", "attrition", "default", "fraud", "grade", "demand_tier"
    ]
}

def detect_column_mappings(df_columns: list) -> dict:
    """Map detected columns from dataset to canonical schema using alias matching."""
    clean_cols = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in df_columns}
    mapping = {}
    
    for canonical, aliases in COLUMN_ALIASES.items():
        matched_original = None
        for alias in aliases:
            if alias in clean_cols:
                matched_original = clean_cols[alias]
                break
        if matched_original:
            mapping[matched_original] = canonical
            
    return mapping


def generate_synthetic_sales_data(days: int = 730, output_path: str = "data/demo_sales.csv") -> pd.DataFrame:
    """
    Generate realistic multi-year enterprise sales dataset with trends,
    seasonality, price elasticity, and regional variations.
    """
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_date = datetime(2024, 1, 1)
    date_list = [start_date + timedelta(days=i) for i in range(days)]
    
    catalog = [
        # Enterprise Software
        {"Product": "Cloud ERP Suite", "Category": "Enterprise Software", "Base_Price": 1250.0, "Base_Vol": 14, "Stock": 420},
        {"Product": "Cybersecurity Pro", "Category": "Enterprise Software", "Base_Price": 850.0, "Base_Vol": 22, "Stock": 550},
        {"Product": "AI Analytics Core", "Category": "Enterprise Software", "Base_Price": 1600.0, "Base_Vol": 11, "Stock": 280},
        {"Product": "DevOps Pipeline Hub", "Category": "Enterprise Software", "Base_Price": 450.0, "Base_Vol": 28, "Stock": 600},
        
        # Hardware & Peripherals
        {"Product": "UltraWide 4K Monitor", "Category": "Hardware", "Base_Price": 620.0, "Base_Vol": 35, "Stock": 750},
        {"Product": "Mechanical Pro Keyboard", "Category": "Hardware", "Base_Price": 140.0, "Base_Vol": 65, "Stock": 1400},
        {"Product": "Ergonomic Dock Station", "Category": "Hardware", "Base_Price": 210.0, "Base_Vol": 48, "Stock": 920},
        {"Product": "Enterprise Server Blade", "Category": "Hardware", "Base_Price": 3400.0, "Base_Vol": 6, "Stock": 130},
        
        # Cloud Services
        {"Product": "Dedicated Compute Node", "Category": "Cloud Services", "Base_Price": 750.0, "Base_Vol": 30, "Stock": 800},
        {"Product": "High-IOPS Storage Tier", "Category": "Cloud Services", "Base_Price": 320.0, "Base_Vol": 55, "Stock": 1200},
        {"Product": "Managed Kubernetes", "Category": "Cloud Services", "Base_Price": 980.0, "Base_Vol": 18, "Stock": 450},
        {"Product": "CDN Edge Accelerator", "Category": "Cloud Services", "Base_Price": 220.0, "Base_Vol": 45, "Stock": 1100},
        
        # Business Supplies
        {"Product": "Conference Audio Bar", "Category": "Business Supplies", "Base_Price": 380.0, "Base_Vol": 25, "Stock": 500},
        {"Product": "Executive Workstation Kit", "Category": "Business Supplies", "Base_Price": 520.0, "Base_Vol": 20, "Stock": 380},
        {"Product": "High-Yield Toner Pack", "Category": "Business Supplies", "Base_Price": 110.0, "Base_Vol": 75, "Stock": 1600}
    ]
    
    regions = {
        "North America": ["New York", "San Francisco", "Chicago", "Austin", "Toronto"],
        "Europe": ["London", "Frankfurt", "Paris", "Amsterdam", "Stockholm"],
        "Asia-Pacific": ["Tokyo", "Singapore", "Sydney", "Seoul", "Bangalore"],
        "Latin America": ["São Paulo", "Mexico City", "Buenos Aires", "Bogota"],
        "Middle East": ["Dubai", "Riyadh", "Abu Dhabi", "Doha"]
    }
    
    customer_segments = ["Enterprise", "Mid-Market", "Small Business"]
    segment_weights = [0.45, 0.35, 0.20]
    
    records = []
    
    for current_date in date_list:
        day_of_year = current_date.timetuple().tm_yday
        day_of_week = current_date.weekday() # 0=Mon, 6=Sun
        month = current_date.month
        
        # Seasonality factors
        # Tech B2B dips slightly on weekends
        weekend_mult = 0.55 if day_of_week in [5, 6] else 1.08
        
        # Q4 surge (October, November, December) budget flush
        quarter_mult = 1.25 if month in [10, 11, 12] else (0.92 if month in [7, 8] else 1.02)
        
        # Long-term business growth trend (+18% annually)
        days_from_start = (current_date - start_date).days
        trend_mult = 1.0 + (0.18 * (days_from_start / 365.0))
        
        # Random daily transactions
        num_transactions = np.random.randint(7, 15)
        
        for _ in range(num_transactions):
            item = np.random.choice(catalog)
            region = np.random.choice(list(regions.keys()), p=[0.38, 0.28, 0.20, 0.08, 0.06])
            city = np.random.choice(regions[region])
            segment = np.random.choice(customer_segments, p=segment_weights)
            
            # Segment volume multiplier
            seg_mult = 1.8 if segment == "Enterprise" else (1.1 if segment == "Mid-Market" else 0.7)
            
            # Discount variation (0% to 22%)
            discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.40, 0.25, 0.20, 0.10, 0.05]))
            
            # Price elasticity: discount gives small volume lift
            promo_lift = 1.0 + (discount * 1.2)
            
            expected_qty = item["Base_Vol"] * weekend_mult * quarter_mult * trend_mult * seg_mult * promo_lift
            # Add realistic Poisson/Normal noise
            quantity = max(1, int(np.random.poisson(lam=max(1.0, expected_qty / 4.0))))
            
            # Unit price with tiny random variance (+/- 3%)
            unit_price = round(item["Base_Price"] * float(np.random.uniform(0.97, 1.03)), 2)
            
            # Total revenue
            revenue = round(quantity * unit_price * (1.0 - discount), 2)
            
            # Current inventory baseline for the item
            stock = item["Stock"]
            
            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Product": item["Product"],
                "Category": item["Category"],
                "Quantity": quantity,
                "Unit_Price": unit_price,
                "Discount": discount,
                "Revenue": revenue,
                "Region": region,
                "City": city,
                "Customer_Segment": segment,
                "Current_Stock": stock
            })
            
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_dataset(uploaded_file_or_path) -> tuple[pd.DataFrame, dict, list]:
    """
    Load dataset from path or uploaded file buffer.
    Returns:
        standardized_df: DataFrame with standardized columns
        quality_info: Dictionary of data metrics
        warnings: List of warning/info messages
    """
    warnings = []
    
    if uploaded_file_or_path is None:
        demo_path = "data/demo_sales.csv"
        if not os.path.exists(demo_path):
            df_raw = generate_synthetic_sales_data(output_path=demo_path)
            warnings.append("Demo synthetic dataset automatically generated with 24 months of data.")
        else:
            df_raw = pd.read_csv(demo_path)
    elif isinstance(uploaded_file_or_path, str):
        if uploaded_file_or_path.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file_or_path)
        else:
            df_raw = pd.read_csv(uploaded_file_or_path)
    else:
        # Streamlit UploadedFile object
        file_name = uploaded_file_or_path.name.lower()
        if file_name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded_file_or_path)
        else:
            df_raw = pd.read_csv(uploaded_file_or_path)

    if df_raw.empty:
        raise ValueError("The provided dataset is completely empty.")

    original_shape = df_raw.shape
    
    # 1. Detect and map standard aliases
    mapping = detect_column_mappings(df_raw.columns)
    df = df_raw.copy()
    
    # Track which original columns mapped to canonical roles
    mapped_canonicals = {v: k for k, v in mapping.items()}

    # 2. Date column resolution or synthesis (guarantees ANY dataset works)
    date_col = None
    if "date" in mapped_canonicals:
        date_col = mapped_canonicals["date"]
    else:
        # Search for any column that looks like a datetime
        for c in df.columns:
            if df[c].dtype == "object" or "int" in str(df[c].dtype):
                try:
                    sample = df[c].dropna().head(20)
                    parsed_sample = pd.to_datetime(sample, errors="coerce")
                    if parsed_sample.notna().sum() >= max(3, int(len(sample) * 0.7)):
                        date_col = c
                        warnings.append(f"Inferred '{c}' as the chronological Date column.")
                        break
                except Exception:
                    continue

    if date_col is not None:
        try:
            df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
            invalid_dates_count = int(df["Date"].isna().sum())
            if invalid_dates_count > 0:
                warnings.append(f"Imputed/fixed {invalid_dates_count} invalid date values.")
                # Forward fill or fill with min date to keep all user rows
                df["Date"] = df["Date"].bfill().ffill()
                if df["Date"].isna().any():
                    df["Date"] = df["Date"].fillna(pd.Timestamp("2024-01-01"))
        except Exception:
            date_col = None

    if date_col is None or "Date" not in df.columns:
        # Synthesize a sequential business date timeline so all analytics, trends, forecasting, and ML work smoothly
        base_start = datetime(2024, 1, 1)
        num_rows = len(df)
        # Distribute rows across realistic calendar days
        days_span = max(30, min(730, num_rows // 2 if num_rows > 60 else num_rows))
        synthesized_dates = [base_start + timedelta(days=int((i / max(1, num_rows - 1)) * days_span)) for i in range(num_rows)]
        df["Date"] = synthesized_dates
        warnings.append("No explicit Date field detected: automatically constructed chronological business timeline so time-series forecasting and seasonal modules function seamlessly.")

    # 3. Product resolution (infer highest-cardinality categorical if missing)
    if "product" in mapped_canonicals:
        df["Product"] = df[mapped_canonicals["product"]].astype(str)
    else:
        # Find best categorical column
        cat_candidates = [c for c in df_raw.columns if c != "Date" and df_raw[c].dtype == "object" and 1 < df_raw[c].nunique() <= 500]
        if cat_candidates:
            best_cat = cat_candidates[0]
            df["Product"] = df[best_cat].astype(str)
            warnings.append(f"Inferred '{best_cat}' as Product SKU identifier.")
        else:
            # Fallback: create cluster or generic SKU names
            df["Product"] = [f"Item_{i % 8 + 1}" for i in range(len(df))]
            warnings.append("No Product column identified; generated standard catalog SKU groups.")

    # 4. Category resolution
    if "category" in mapped_canonicals:
        df["Category"] = df[mapped_canonicals["category"]].astype(str)
    else:
        # Look for another categorical column with low cardinality (2 to 25 unique values)
        other_cats = [c for c in df_raw.columns if c not in ["Date", "Product"] and df_raw[c].dtype == "object" and 1 < df_raw[c].nunique() <= 25]
        if other_cats:
            best_other = other_cats[0]
            df["Category"] = df[best_other].astype(str)
            warnings.append(f"Inferred '{best_other}' as Category classification.")
        else:
            # Map based on product prefix or default
            df["Category"] = df["Product"].apply(lambda p: f"Category {str(p)[:3].upper()}" if len(str(p)) >= 3 else "General")

    # 5. Quantity, Unit Price, and Revenue resolution
    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    
    if "quantity" in mapped_canonicals:
        df["Quantity"] = pd.to_numeric(df[mapped_canonicals["quantity"]], errors="coerce").fillna(1).clip(lower=1).astype(int)
    else:
        int_cols = [c for c in numeric_cols if (df_raw[c] % 1 == 0).all() and (df_raw[c] > 0).all() and df_raw[c].max() <= 50000]
        if int_cols:
            df["Quantity"] = pd.to_numeric(df[int_cols[0]], errors="coerce").fillna(1).clip(lower=1).astype(int)
            warnings.append(f"Inferred '{int_cols[0]}' as transaction Quantity.")
        else:
            df["Quantity"] = 1

    if "revenue" in mapped_canonicals:
        df["Revenue"] = pd.to_numeric(df[mapped_canonicals["revenue"]], errors="coerce").fillna(50.0).clip(lower=0.1)
    elif "unit_price" in mapped_canonicals and "Quantity" in df.columns:
        p = pd.to_numeric(df[mapped_canonicals["unit_price"]], errors="coerce").fillna(25.0)
        df["Revenue"] = (p * df["Quantity"]).clip(lower=0.1)
    else:
        float_cols = [c for c in numeric_cols if c not in [mapped_canonicals.get("quantity", ""), "Quantity"]]
        if float_cols:
            df["Revenue"] = pd.to_numeric(df[float_cols[0]], errors="coerce").fillna(100.0).clip(lower=0.1)
            warnings.append(f"Inferred '{float_cols[0]}' as Revenue.")
        else:
            df["Revenue"] = (df["Quantity"] * 50.0).astype(float)

    if "unit_price" in mapped_canonicals:
        df["Unit_Price"] = pd.to_numeric(df[mapped_canonicals["unit_price"]], errors="coerce").fillna(10.0).clip(lower=0.01)
    else:
        df["Unit_Price"] = (df["Revenue"] / df["Quantity"].replace(0, 1)).clip(lower=0.01).round(2)

    if "discount" in mapped_canonicals:
        disc = pd.to_numeric(df[mapped_canonicals["discount"]], errors="coerce").fillna(0.0)
        # Normalize if given as percentage 0-100
        if disc.max() > 1.0:
            disc = disc / 100.0
        df["Discount"] = disc.clip(0.0, 0.9)
    else:
        df["Discount"] = 0.0

    if "region" in mapped_canonicals:
        df["Region"] = df[mapped_canonicals["region"]].astype(str)
    else:
        geo_candidates = [c for c in df_raw.columns if c not in ["Product", "Category", "Date"] and df_raw[c].dtype == "object" and 1 < df_raw[c].nunique() <= 15]
        if geo_candidates:
            df["Region"] = df[geo_candidates[0]].astype(str)
            warnings.append(f"Inferred '{geo_candidates[0]}' as Region.")
        else:
            df["Region"] = "Global"

    if "city" in mapped_canonicals:
        df["City"] = df[mapped_canonicals["city"]].astype(str)
    else:
        df["City"] = "Metro Hub"

    if "customer_segment" in mapped_canonicals:
        df["Customer_Segment"] = df[mapped_canonicals["customer_segment"]].astype(str)
    else:
        seg_candidates = [c for c in df_raw.columns if c not in ["Product", "Category", "Region", "Date"] and df_raw[c].dtype == "object" and 1 < df_raw[c].nunique() <= 8]
        if seg_candidates:
            df["Customer_Segment"] = df[seg_candidates[0]].astype(str)
        else:
            df["Customer_Segment"] = "Commercial"

    if "current_stock" in mapped_canonicals:
        df["Current_Stock"] = pd.to_numeric(df[mapped_canonicals["current_stock"]], errors="coerce").fillna(100).astype(int)
    else:
        prod_vols = df.groupby("Product")["Quantity"].transform("sum")
        df["Current_Stock"] = (prod_vols * 0.3).clip(lower=50).astype(int)

    # Detect potential classification targets in the dataset (for ML Classification)
    detected_targets = []
    if "target_label" in mapped_canonicals:
        detected_targets.append(mapped_canonicals["target_label"])
    for col in df_raw.columns:
        if col not in detected_targets and col != "Date":
            n_unique = df_raw[col].nunique()
            if 2 <= n_unique <= 12:
                detected_targets.append(col)

    # Clean dates and ensure proper sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    quality_info = {
        "original_rows": original_shape[0],
        "original_cols": original_shape[1],
        "clean_rows": len(df),
        "clean_cols": len(df.columns),
        "start_date": df["Date"].min().strftime("%Y-%m-%d"),
        "end_date": df["Date"].max().strftime("%Y-%m-%d"),
        "date_span_days": max(1, (df["Date"].max() - df["Date"].min()).days),
        "total_revenue": float(df["Revenue"].sum()),
        "total_quantity": int(df["Quantity"].sum()),
        "num_products": int(df["Product"].nunique()),
        "num_categories": int(df["Category"].nunique()),
        "num_regions": int(df["Region"].nunique()),
        "detected_targets": detected_targets,
        "original_columns": list(df_raw.columns)
    }

    return df, quality_info, warnings
