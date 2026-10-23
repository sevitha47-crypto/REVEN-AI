"""
Automated Business Report generation module.
Compiles executive PDF reports using ReportLab containing KPIs, model benchmarks,
inventory risk tables, and strategic recommendations.
"""

import io
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

def generate_pdf_business_report(
    quality_info: dict,
    kpi_summary: dict,
    model_results_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    recommendations: list[dict]
) -> bytes:
    """
    Construct a downloadable multi-page executive PDF report.
    Returns binary PDF byte buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Brand Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    
    table_text_style = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )
    
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("REVENAI — SALES PREDICTION AND ANALYTICS SYSTEM", title_style))
    story.append(Paragraph(f"Executive Decision Support & Intelligence Report — Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S UTC')}", subtitle_style))
    story.append(Spacer(1, 8))
    
    # Section 1: Executive KPI Overview
    story.append(Paragraph("1. Commercial Performance Overview", h2_style))
    
    kpi_data = [
        [
            Paragraph("<b>Metric</b>", table_header_style),
            Paragraph("<b>Value</b>", table_header_style),
            Paragraph("<b>Context / Baseline</b>", table_header_style)
        ],
        [
            Paragraph("Total Revenue", table_text_style),
            Paragraph(f"${kpi_summary.get('total_revenue', 0):,.2f}", table_text_style),
            Paragraph("Aggregated commercial volume", table_text_style)
        ],
        [
            Paragraph("Total Units Sold", table_text_style),
            Paragraph(f"{kpi_summary.get('total_quantity', 0):,}", table_text_style),
            Paragraph("Across all catalog lines", table_text_style)
        ],
        [
            Paragraph("Average Order Value (AOV)", table_text_style),
            Paragraph(f"${kpi_summary.get('avg_order_value', 0):,.2f}", table_text_style),
            Paragraph("Mean gross revenue per sale", table_text_style)
        ],
        [
            Paragraph("Active Product SKUs", table_text_style),
            Paragraph(f"{kpi_summary.get('num_products', 0)}", table_text_style),
            Paragraph(f"Spread over {kpi_summary.get('num_categories', 0)} commercial categories", table_text_style)
        ],
        [
            Paragraph("Data Quality Score", table_text_style),
            Paragraph(f"{quality_info.get('quality_score', 98.5)}%", table_text_style),
            Paragraph("Validated data pipeline score", table_text_style)
        ]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[150, 130, 250])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white])
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))
    
    # Section 2: Machine Learning Benchmark Evaluation
    story.append(Paragraph("2. Predictive Machine Learning Evaluation", h2_style))
    story.append(Paragraph("Models were trained using an 80/20 chronological past-to-future holdout to prevent lookahead bias.", body_style))
    story.append(Spacer(1, 6))
    
    ml_data = [[
        Paragraph("<b>Model Name</b>", table_header_style),
        Paragraph("<b>MAE</b>", table_header_style),
        Paragraph("<b>RMSE</b>", table_header_style),
        Paragraph("<b>R² Score</b>", table_header_style),
        Paragraph("<b>Status</b>", table_header_style)
    ]]
    
    for _, row in model_results_df.iterrows():
        ml_data.append([
            Paragraph(str(row["Model"]), table_text_style),
            Paragraph(f"{row['MAE']:.3f}", table_text_style),
            Paragraph(f"{row['RMSE']:.3f}", table_text_style),
            Paragraph(f"{row['R2']:.4f}", table_text_style),
            Paragraph(str(row.get("Status", "Evaluated")), table_text_style)
        ])
        
    ml_table = Table(ml_data, colWidths=[160, 80, 80, 80, 130])
    ml_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white])
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 14))
    
    # Section 3: Priority Inventory Risks
    story.append(Paragraph("3. Critical Inventory & Replenishment Risks", h2_style))
    
    inv_top = inv_df.head(6)
    inv_data = [[
        Paragraph("<b>Product</b>", table_header_style),
        Paragraph("<b>Current Stock</b>", table_header_style),
        Paragraph("<b>Days Coverage (DOIR)</b>", table_header_style),
        Paragraph("<b>Risk Status</b>", table_header_style),
        Paragraph("<b>Reorder Units</b>", table_header_style)
    ]]
    
    for _, row in inv_top.iterrows():
        inv_data.append([
            Paragraph(str(row["Product"]), table_text_style),
            Paragraph(f"{int(row['Current_Stock']):,}", table_text_style),
            Paragraph(f"{row['DOIR_Days']:.1f} days", table_text_style),
            Paragraph(str(row["Risk_Status"]), table_text_style),
            Paragraph(f"{int(row['Recommended_Reorder_Units']):,} units", table_text_style)
        ])
        
    inv_table = Table(inv_data, colWidths=[170, 85, 100, 95, 80])
    inv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white])
    ]))
    story.append(inv_table)
    story.append(Spacer(1, 14))
    
    # Section 4: Actionable Business Recommendations
    story.append(Paragraph("4. Prescriptive Business Recommendations", h2_style))
    
    rec_data = [[
        Paragraph("<b>Category</b>", table_header_style),
        Paragraph("<b>Priority</b>", table_header_style),
        Paragraph("<b>Recommendation</b>", table_header_style),
        Paragraph("<b>Empirical Rationale</b>", table_header_style)
    ]]
    
    for rec in recommendations[:5]:
        rec_data.append([
            Paragraph(rec["category"], table_text_style),
            Paragraph(rec["priority"], table_text_style),
            Paragraph(rec["title"] + "<br/>" + rec["recommendation"], table_text_style),
            Paragraph(rec["reason"], table_text_style)
        ])
        
    rec_table = Table(rec_data, colWidths=[100, 60, 180, 190])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white])
    ]))
    story.append(rec_table)
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
