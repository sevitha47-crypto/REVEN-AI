"""
RevenAI Strategic Decision Engine: AI Business & Marketing Strategies for Low Stock and Sales Prediction.
Integrates Gemini 3.8 Flash with robust algorithmic heuristic fallback.
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def calculate_stock_health(
    predicted_units: float,
    current_stock: float,
    unit_price: float,
    discount: float,
    historical_daily_avg: float = 5.0
) -> dict:
    """
    Computes stock coverage ratio, deficit/surplus, and stockout risk metrics.
    """
    predicted_units = max(0.1, float(predicted_units))
    current_stock = max(0, float(current_stock))
    unit_price = max(1.0, float(unit_price))
    
    # Coverage in terms of predicted inferenced demand
    coverage_ratio = current_stock / predicted_units
    unit_deficit = max(0.0, predicted_units - current_stock)
    revenue_at_risk = unit_deficit * unit_price * (1.0 - discount)
    
    # Estimate days until stock exhaustion
    daily_burn_rate = max(0.2, historical_daily_avg if historical_daily_avg > 0 else predicted_units / 7.0)
    days_to_stockout = current_stock / daily_burn_rate
    
    # Health status determination
    if current_stock == 0:
        status = "OUT OF STOCK"
        severity = "critical"
        badge_class = "badge-high"
        headline = "Immediate Stockout: 0 Units on Hand"
    elif current_stock < predicted_units * 0.5 or days_to_stockout < 5:
        status = "CRITICAL LOW STOCK"
        severity = "critical"
        badge_class = "badge-high"
        headline = f"Severe Stock Deficit ({int(current_stock):,} units vs {int(round(predicted_units)):,} predicted demand)"
    elif current_stock < predicted_units or days_to_stockout < 14:
        status = "LOW STOCK WARNING"
        severity = "warning"
        badge_class = "badge-medium"
        headline = f"Low Inventory Reserve ({int(current_stock):,} units, ~{int(round(days_to_stockout))} days coverage)"
    elif current_stock < predicted_units * 1.5:
        status = "MODERATE STOCK"
        severity = "moderate"
        badge_class = "badge-low"
        headline = f"Adequate Stock Buffer ({int(current_stock):,} units on hand)"
    else:
        status = "HEALTHY STOCK"
        severity = "healthy"
        badge_class = "badge-low"
        headline = f"Healthy Inventory Buffer ({int(current_stock):,} units on hand)"

    return {
        "status": status,
        "severity": severity,
        "badge_class": badge_class,
        "headline": headline,
        "is_low_stock": severity in ["critical", "warning"],
        "coverage_ratio": round(coverage_ratio, 2),
        "coverage_pct": round(coverage_ratio * 100, 1),
        "unit_deficit": int(round(unit_deficit)),
        "current_stock": int(current_stock),
        "predicted_units": int(round(predicted_units)),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "days_to_stockout": round(days_to_stockout, 1)
    }


def get_heuristic_strategies(
    product: str,
    category: str,
    region: str,
    customer_segment: str,
    stock_info: dict,
    unit_price: float,
    discount: float
) -> dict:
    """
    High-craft, domain-driven heuristic business and marketing strategies
    designed specifically for low stock conditions in sales prediction.
    """
    stock = stock_info["current_stock"]
    pred = stock_info["predicted_units"]
    deficit = stock_info["unit_deficit"]
    days = stock_info["days_to_stockout"]
    at_risk_rev = stock_info["revenue_at_risk"]
    
    marketing_strategies = [
        {
            "tactic": "Paid Ad Throttling & Budget Shift",
            "action": f"Immediately pause direct-response PPC (Google Shopping/Meta) for {product}.",
            "detail": f"Reallocate current ad spend to high-inventory alternative items in '{category}'. Running paid ads on an SKU with only {stock} units wastes acquisition budget and results in bounce clicks once exhausted.",
            "impact": "Eliminate Ad Spend Waste ($1.5k–$4k saved/wk)",
            "icon": "🛑"
        },
        {
            "tactic": "Scarcity & Ethical FOMO Conversion",
            "action": f"Activate live stock badge: 'Only {stock} units remaining in stock'.",
            "detail": f"Highlight real-time inventory scarcity on digital storefronts and sales quotes. This accelerates buyer decision-making while eliminating the need for discounts ({discount*100:.0f}% discount should be dropped to 0%).",
            "impact": "Gross Margin Protection & Accelerated Checkout",
            "icon": "🔥"
        },
        {
            "tactic": "Pre-Order & VIP Priority Waitlist",
            "action": f"Deploy a 'Pre-Order for Next Inbound Batch' or 'Join VIP Restock Alert' CTA.",
            "detail": f"When stock drops below buffer, allow {customer_segment} customers to reserve upcoming allocations with a guaranteed ship date or exclusive priority perk, locking in cash flow before stock arrivals.",
            "impact": "Retain 70–85% of Unmet Customer Demand",
            "icon": "📋"
        },
        {
            "tactic": "Smart Catalog Redirection & Cross-Selling",
            "action": f"Configure automatic 'Recommended Substitutes' module for {category}.",
            "detail": f"Feature complementary items and equivalent models prominently on the {product} product page and checkout flow to capture demand that would otherwise churn to competitors.",
            "impact": "Revenue Retention across Related SKUs",
            "icon": "🔀"
        }
    ]
    
    business_strategies = [
        {
            "tactic": "Emergency Replenishment & Expedited Freight",
            "action": f"Issue an immediate expedited Purchase Order for {max(deficit + 50, 100)} units.",
            "detail": f"Forecast indicates demand will outstrip on-hand units in ~{days:.0f} days. Contract an air-freight split shipment from tier-1 suppliers to bridge the gap before the standard bulk shipment arrives.",
            "impact": f"Protect ${at_risk_rev:,.0f} in Potential Lost Revenue",
            "icon": "✈️"
        },
        {
            "tactic": "Account-Based Customer Rationing",
            "action": f"Cap spot order volume and prioritize {customer_segment} contract agreements.",
            "detail": f"Implement a maximum order quantity (e.g., 2–5 units per account) to prevent a single buyer or wholesale reseller from depleting the entire {stock}-unit inventory, preserving satisfaction for key accounts.",
            "impact": "Protect Long-Term Client Relationships",
            "icon": "🛡️"
        },
        {
            "tactic": "Regional Inventory Rebalancing",
            "action": f"Initiate inter-depot warehouse transfer into {region}.",
            "detail": f"Audit secondary distribution hubs for surplus inventory in {product}. Moving stock into {region} neutralizes local deficits faster than waiting for international manufacturing lead times.",
            "impact": "Reduce Fulfillment Lead Time by 40–60%",
            "icon": "🚚"
        },
        {
            "tactic": "Dynamic Margin & Price Optimization",
            "action": f"Freeze all discretionary discounting and test price elasticity (+5% to +10%).",
            "detail": f"With high demand velocity and scarce units ({stock} available vs {pred} projected), price elasticity is inelastic. Raising the price to ${unit_price * 1.08:.2f} maximizes gross margin per scarce unit.",
            "impact": "+8% to +12% Net Margin per Remaining Unit",
            "icon": "📈"
        }
    ]

    marketing_copy = {
        "email_subject": f"⚡ Low Stock Alert: Limited remaining allocation for {product}",
        "email_body": f"Dear Customer,\n\nDue to strong commercial demand in {region}, inventory for the {product} is currently down to just {stock} units. If you are preparing your order, we recommend confirming allocation today to avoid backorder delays.\n\nReserve your units or explore our guaranteed pre-order options here.",
        "storefront_banner": f"⚠️ High Demand Alert: Only {stock} units left in stock. Orders ship on a first-come, first-served basis.",
        "sales_rep_pitch": f"\"Hi [Client], I noticed you were evaluating {product}. Our warehouse shows only {stock} units left in {region} against strong incoming orders. I can lock in your allocation today before we transition to backorder status.\""
    }

    return {
        "source": "RevenAI Strategic Engine (Algorithmic Heuristic)",
        "marketing_strategies": marketing_strategies,
        "business_strategies": business_strategies,
        "marketing_copy": marketing_copy,
        "key_takeaway": f"Immediate supply-demand mismatch for {product}: Pivot marketing spend from acquisition to waitlist capture, freeze discounts, and expedite emergency replenishment to safeguard ${at_risk_rev:,.0f} in gross revenue."
    }


def generate_ai_low_stock_strategy(
    product: str,
    category: str,
    region: str,
    customer_segment: str,
    stock_info: dict,
    unit_price: float,
    discount: float,
    use_gemini: bool = True
) -> dict:
    """
    Orchestrates AI strategy generation. Attempts Gemini 3.8 Flash first,
    and cleanly falls back to domain heuristic engine if API is unavailable or offline.
    """
    heuristic_fallback = get_heuristic_strategies(
        product=product,
        category=category,
        region=region,
        customer_segment=customer_segment,
        stock_info=stock_info,
        unit_price=unit_price,
        discount=discount
    )
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not use_gemini or not api_key:
        return heuristic_fallback

    # Build targeted prompt for Gemini
    prompt = f"""
You are the Chief Commercial Officer and Supply Chain Strategist of RevenAI, an elite enterprise sales intelligence system.
Analyze the following low-stock sales prediction scenario and provide actionable business and marketing strategies.

SCENARIO METRICS:
- Product: {product} (Category: {category})
- Commercial Region: {region}
- Target Customer Segment: {customer_segment}
- Unit Price: ${unit_price:.2f} (Current Discount: {discount*100:.0f}%)
- Current Available Stock: {stock_info['current_stock']} units
- Predicted Demand for Target Horizon: {stock_info['predicted_units']} units
- Projected Unit Deficit: {stock_info['unit_deficit']} units
- Days to Exhaustion: {stock_info['days_to_stockout']} days
- Estimated Revenue at Risk: ${stock_info['revenue_at_risk']:,.2f}

Provide a structured, executive-level JSON response with:
1. "key_takeaway": A concise 2-sentence executive summary.
2. "marketing_strategies": A list of 4 specific marketing tactics for when units are low/stockout imminent (e.g. ad spend shifting, scarcity messaging, pre-order/waitlist capture, upsell/cross-sell to high-stock substitutes, discount elimination). Each must have "tactic", "action", "detail", "impact", "icon".
3. "business_strategies": A list of 4 specific operational/supply chain business tactics (e.g. emergency replenishment PO, customer rationing for high-LTV tiers, warehouse rebalancing, price elasticity adjustment). Each must have "tactic", "action", "detail", "impact", "icon".
4. "marketing_copy": An object containing "email_subject", "email_body", "storefront_banner", and "sales_rep_pitch".

Respond ONLY with valid JSON. No conversational preamble, no markdown backticks.
"""

    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                data = response.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean possible markdown blocks if returned
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                
                parsed = json.loads(raw_text)
                model_label = "Gemini 3.8 Flash" if "3.8" in model_name else "Gemini 3.6 Flash"
                parsed["source"] = f"RevenAI Intelligence Engine ({model_label} Live)"
                return parsed
            else:
                logger.warning(f"Model {model_name} returned status {response.status_code}, trying next model or fallback.")
        except Exception as e:
            logger.warning(f"Model {model_name} call failed ({e}), trying next model or fallback.")

    return heuristic_fallback
