from datetime import date, datetime
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from model import (
    score_expense_pressure,
    score_income_volatility,
    score_payment_consistency,
    score_positive_trend,
)

try:
    from streamlit_lottie import st_lottie
except Exception:  # pragma: no cover - optional UI dependency
    st_lottie = None

BASE_DIR = Path(__file__).parent
FAVICON_PATH = BASE_DIR / "favicon.ico"
PAGE_ICON = str(FAVICON_PATH) if FAVICON_PATH.exists() else "🏦"

st.set_page_config(
    page_title="Stelvora Dashboard",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = BASE_DIR / "loan_data.csv"
SCORES_PATH = BASE_DIR / "loan_scores.csv"
RISK_ORDER = ["Green", "Yellow", "Orange", "Red"]
RISK_COLORS = {
    "Green": "#67d17a",
    "Yellow": "#f1d36b",
    "Orange": "#f5b13c",
    "Red": "#8f7cf7",
}
THEME_BG = "#222223"
SIDEBAR_BG = "#1d1d1f"
CARD_BG = "#262628"
CARD_BORDER = "#333337"
TEXT_PRIMARY = "#f2f2f3"
TEXT_MUTED = "#9a9aa0"
ACCENT_PURPLE = "#8f7cf7"
ACCENT_ORANGE = "#f5b13c"
ACCENT_GREEN = "#67d17a"
ACCENT_RED = "#ff6b6b"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
BORROWER_TYPE_LABELS = {
    "small_trader": "MSME",
    "farmer": "Farmer",
    "gig_worker": "Gig Worker",
    "salaried": "Salaried",
}


def apply_dark_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: #17181a;
                color: {TEXT_PRIMARY};
                font-family: "Inter", "Roboto", "Segoe UI", sans-serif;
            }}

            [data-testid="stHeader"] {{
                background-color: transparent !important;
                backdrop-filter: none !important;
                border-bottom: none !important;
                z-index: 30 !important;
            }}

            .block-container {{
                padding-top: 3.6rem;
                padding-bottom: 0.8rem;
                padding-left: 0.95rem;
                padding-right: 0.95rem;
                max-width: 1480px;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #33206f 0%, #1f2558 32%, #111827 100%);
                border-right: 1px solid {CARD_BORDER};
            }}

            [data-testid="stSidebar"] > div:first-child {{
                background: linear-gradient(180deg, #33206f 0%, #1f2558 32%, #111827 100%);
            }}

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div {{
                color: {TEXT_PRIMARY};
            }}

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="popover"] > div,
            [data-testid="stSidebar"] [data-baseweb="tag"] {{
                background: rgba(19, 20, 27, 0.78);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
            }}

            [data-testid="stSidebar"] button {{
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                color: {TEXT_PRIMARY};
                min-height: 2.55rem;
            }}

            [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {{
                background: {ACCENT_PURPLE};
                border-color: {ACCENT_PURPLE};
                box-shadow: 0 0 0 6px rgba(143, 124, 247, 0.16);
            }}

            [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {{
                background: linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_ORANGE});
            }}

            .hero-panel,
            .metric-card,
            .profile-card {{
                background: #232426;
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 15px;
                box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 12px 32px rgba(0, 0, 0, 0.26);
            }}

            .hero-panel {{
                padding: 0;
                margin-bottom: 1rem;
                background: transparent;
                border: none;
                box-shadow: none;
            }}

            .dashboard-header {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1.15rem;
                margin-top: 0.2rem;
                position: relative;
                z-index: 24;
            }}

            .header-eyebrow {{
                color: {TEXT_MUTED};
                font-size: 0.96rem;
                margin-bottom: 0.15rem;
            }}

            .hero-title {{
                font-size: 2.3rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                color: {TEXT_PRIMARY};
                line-height: 1.08;
                margin-bottom: 0.2rem;
            }}

            .hero-subtitle {{
                color: {TEXT_MUTED};
                font-size: 0.98rem;
                margin-bottom: 0;
            }}

            .header-actions {{
                display: flex;
                align-items: center;
                gap: 0.65rem;
                position: relative;
                z-index: 26;
                margin-top: 0.2rem;
                padding-right: 0.15rem;
                flex-wrap: wrap;
            }}

            .header-pill {{
                background: #2a2a2c;
                border: 1px solid {CARD_BORDER};
                color: {TEXT_PRIMARY};
                border-radius: 12px;
                padding: 0.65rem 0.9rem;
                font-size: 0.88rem;
                font-weight: 600;
                position: relative;
                z-index: 27;
                pointer-events: auto;
            }}

            .metric-card {{
                padding: 0;
                min-height: 156px;
                background: transparent;
                border: none;
                box-shadow: none;
            }}

            .metric-top {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.9rem;
            }}

            .metric-icon {{
                font-size: 0.95rem;
                opacity: 0.95;
            }}

            .metric-label {{
                color: {TEXT_MUTED};
                text-transform: none;
                letter-spacing: 0;
                font-size: 0.98rem;
                margin-bottom: 0;
            }}

            .metric-value {{
                color: {TEXT_PRIMARY};
                font-size: 2.25rem;
                font-weight: 700;
                margin-bottom: 0.75rem;
            }}

            .metric-footnote {{
                color: {TEXT_MUTED};
                font-size: 0.9rem;
            }}

            .metric-progress-track {{
                width: 100%;
                height: 4px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.06);
                overflow: hidden;
                margin: 0.05rem 0 0.72rem 0;
            }}

            .metric-progress-fill {{
                height: 100%;
                border-radius: 999px;
            }}

            .section-label {{
                color: {TEXT_PRIMARY};
                font-size: 1.08rem;
                font-weight: 700;
                margin: 0.4rem 0 0.75rem 0;
                letter-spacing: 0.01em;
            }}

            .profile-card {{
                padding: 1.2rem 1.2rem 1rem 1.2rem;
            }}

            .comparison-card {{
                background: {CARD_BG};
                border: 1px solid {CARD_BORDER};
                border-radius: 18px;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
                padding: 1.2rem;
                min-height: 245px;
            }}

            .comparison-title {{
                color: #f8fafc;
                font-size: 1.08rem;
                font-weight: 700;
                margin-bottom: 0.9rem;
            }}

            .comparison-item {{
                color: #dbe5f0;
                font-size: 0.96rem;
                margin-bottom: 0.55rem;
            }}

            .timeline-badge {{
                background: rgba(245, 177, 60, 0.12);
                border: 1px solid rgba(245, 177, 60, 0.22);
                color: #fbe7ba;
                border-radius: 999px;
                padding: 0.6rem 0.95rem;
                display: inline-block;
                font-size: 0.95rem;
                font-weight: 600;
                margin-top: 0.4rem;
            }}

            .profile-item-label {{
                color: #8ca0b8;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-top: 0.55rem;
            }}

            .profile-item-value {{
                color: {TEXT_PRIMARY};
                font-size: 1.02rem;
                margin-top: 0.15rem;
            }}

            .sidebar-brand {{
                display: flex;
                align-items: center;
                gap: 0.8rem;
                margin: 0.15rem 0 0.8rem 0;
            }}

            .sidebar-brand-title {{
                color: {TEXT_PRIMARY};
                font-size: 1.42rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                margin: 0;
            }}

            .sidebar-brand-subtitle {{
                color: {TEXT_MUTED};
                font-size: 0.78rem;
                letter-spacing: 0.16em;
                margin-top: 0.12rem;
                text-transform: uppercase;
            }}

            .section-card {{
                background: #232426;
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 15px;
                box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 12px 32px rgba(0,0,0,0.26);
                padding: 1.05rem 1.12rem 0.85rem 1.12rem;
            }}

            .section-card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-top: 0.15rem;
                margin-bottom: 0.8rem;
                position: relative;
                z-index: 8;
            }}

            .section-card-title {{
                color: {TEXT_PRIMARY};
                font-size: 1rem;
                font-weight: 700;
            }}

            .section-card-chip {{
                color: {TEXT_MUTED};
                font-size: 0.8rem;
                border: 1px solid {CARD_BORDER};
                background: #2a2a2c;
                border-radius: 10px;
                padding: 0.45rem 0.7rem;
            }}

            .risk-pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin: 0 0 0.45rem 0;
            }}

            .risk-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.34rem 0.62rem;
                background: #2c2c2f;
                border: 1px solid {CARD_BORDER};
                color: {TEXT_MUTED};
                font-size: 0.78rem;
            }}

            .risk-pill-dot {{
                width: 8px;
                height: 8px;
                border-radius: 999px;
                display: inline-block;
            }}

            .watchlist-shell {{
                background: #232426;
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 15px;
                box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 12px 32px rgba(0,0,0,0.26);
                overflow: hidden;
            }}

            .watchlist-table {{
                width: 100%;
                border-collapse: collapse;
            }}

            .watchlist-table thead th {{
                text-align: left;
                font-size: 0.8rem;
                color: {TEXT_MUTED};
                font-weight: 600;
                padding: 0.95rem 1rem;
                border-bottom: 1px solid {CARD_BORDER};
                background: #2a2a2c;
            }}

            .watchlist-table tbody td {{
                padding: 0.95rem 1rem;
                border-bottom: 1px solid rgba(255,255,255,0.04);
                color: {TEXT_PRIMARY};
                font-size: 0.9rem;
                vertical-align: middle;
            }}

            .watchlist-table tbody tr:hover {{
                background: rgba(255,255,255,0.02);
            }}

            .watchlist-badge {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 74px;
                padding: 0.3rem 0.62rem;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
            }}

            .watchlist-prob {{
                min-width: 180px;
            }}

            .watchlist-prob-label {{
                color: {TEXT_MUTED};
                font-size: 0.76rem;
                margin-bottom: 0.35rem;
            }}

            .watchlist-prob-track {{
                width: 100%;
                height: 8px;
                background: #1f1f21;
                border-radius: 999px;
                overflow: hidden;
                border: 1px solid rgba(255,255,255,0.05);
            }}

            .watchlist-prob-fill {{
                height: 100%;
                border-radius: 999px;
            }}

            .watchlist-action {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.45rem 0.8rem;
                border-radius: 10px;
                font-size: 0.82rem;
                font-weight: 700;
                color: #111318;
                background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_ORANGE});
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
            }}

            .watchlist-scroll {{
                overflow-x: auto;
            }}

            [data-testid="stPlotlyChart"] {{
                position: relative;
                z-index: 2;
                margin-top: 0.2rem;
                margin-bottom: 0.9rem;
            }}

            [data-testid="stPlotlyChart"] > div {{
                overflow: visible !important;
            }}

            .js-plotly-plot .plot-container,
            .js-plotly-plot .svg-container {{
                overflow: visible !important;
            }}

            .scan-indicator {{
                position: fixed;
                right: 1.2rem;
                bottom: 1.1rem;
                z-index: 999;
                display: inline-flex;
                align-items: center;
                gap: 0.6rem;
                padding: 0.7rem 0.95rem;
                border-radius: 999px;
                background: rgba(16, 17, 21, 0.82);
                border: 1px solid rgba(143, 124, 247, 0.28);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
                backdrop-filter: blur(8px);
            }}

            .scan-dot {{
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: {ACCENT_PURPLE};
                box-shadow: 0 0 0 0 rgba(143, 124, 247, 0.8);
                animation: scanPulse 1.8s infinite;
            }}

            .scan-text {{
                color: {TEXT_PRIMARY};
                font-size: 0.84rem;
                letter-spacing: 0.02em;
            }}

            @keyframes scanPulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(143,124,247,0.72); }}
                70% {{ box-shadow: 0 0 0 12px rgba(143,124,247,0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(143,124,247,0); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading Stelvora portfolio...")
def load_portfolio() -> pd.DataFrame:
    loan_data = pd.read_csv(DATA_PATH)
    loan_scores = pd.read_csv(SCORES_PATH)
    portfolio = loan_data.merge(loan_scores, on="borrower_id", how="inner")
    portfolio["borrower_segment"] = portfolio["borrower_type"].map(BORROWER_TYPE_LABELS).fillna(
        portfolio["borrower_type"]
    )
    portfolio["risk_bucket"] = pd.Categorical(
        portfolio["risk_bucket"],
        categories=RISK_ORDER,
        ordered=True,
    )
    portfolio["portfolio_status"] = portfolio["risk_bucket"].apply(
        lambda bucket: "Stressed" if bucket in ["Orange", "Red"] else "Healthy"
    )
    return portfolio.sort_values("loan_health_score", ascending=True).reset_index(drop=True)


def format_indian_number(value: float) -> str:
    rounded = int(round(value))
    sign = "-" if rounded < 0 else ""
    digits = str(abs(rounded))
    if len(digits) <= 3:
        return f"{sign}{digits}"
    last_three = digits[-3:]
    remaining = digits[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    return f"{sign}{','.join(parts + [last_three])}"


def format_currency(value: float) -> str:
    return f"₹{format_indian_number(value)}"


def format_compact_inr(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if absolute >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} Lakh"
    return format_currency(value)


def format_pdf_currency(value: float) -> str:
    return f"Rs. {format_indian_number(value)}"


def format_pdf_compact_inr(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_00_00_000:
        return f"Rs. {value / 1_00_00_000:.2f} Cr"
    if absolute >= 1_00_000:
        return f"Rs. {value / 1_00_000:.2f} Lakh"
    return format_pdf_currency(value)


def get_intervention_cutoff(risk_sensitivity: float) -> float:
    return round(30 + (risk_sensitivity * 40), 1)


def build_intervention_watchlist(df: pd.DataFrame, risk_sensitivity: float) -> pd.DataFrame:
    score_cutoff = get_intervention_cutoff(risk_sensitivity)
    watchlist = df.loc[df["loan_health_score"] <= score_cutoff].copy()
    if watchlist.empty:
        return df.nsmallest(10, "loan_health_score").copy()
    return watchlist.sort_values("loan_health_score", ascending=True).copy()


def filter_portfolio_by_segment(df: pd.DataFrame, selected_segments: list[str]) -> pd.DataFrame:
    if not selected_segments:
        return df.iloc[0:0].copy()
    return (
        df.loc[df["borrower_segment"].isin(selected_segments)]
        .sort_values("loan_health_score", ascending=True)
        .reset_index(drop=True)
    )


def build_dashboard_summary(df: pd.DataFrame, risk_sensitivity: float) -> dict[str, float | int | str]:
    score_cutoff = get_intervention_cutoff(risk_sensitivity)
    stressed_df = df.loc[df["loan_health_score"] <= score_cutoff].copy()
    top_10 = (
        stressed_df if not stressed_df.empty else df.nsmallest(10, "loan_health_score")
    ).sort_values("loan_health_score", ascending=True).head(10)

    total_portfolio_value = float(df["loan_amount"].sum())
    stressed_assets_detected = int(len(stressed_df))
    potential_default_value = float(stressed_df["loan_amount"].sum()) if not stressed_df.empty else 0.0
    predicted_recovery_savings = (potential_default_value * 0.70) + (stressed_assets_detected * 8000)

    segments = sorted(df["borrower_segment"].dropna().unique().tolist())
    all_segments = sorted(BORROWER_TYPE_LABELS.values())
    if not segments or segments == all_segments:
        filter_label = "All Borrowers"
    else:
        filter_label = ", ".join(segments)

    return {
        "filtered_df": df,
        "stressed_df": stressed_df,
        "top_10_df": top_10,
        "score_cutoff": score_cutoff,
        "total_loans": int(len(df)),
        "portfolio_health_score": round(float(df["loan_health_score"].mean()), 1) if not df.empty else 0.0,
        "total_portfolio_value": total_portfolio_value,
        "stressed_assets_detected": stressed_assets_detected,
        "potential_default_value": potential_default_value,
        "predicted_recovery_savings": predicted_recovery_savings,
        "filter_label": filter_label,
        "timestamp": datetime.now().strftime("%d %b %Y %I:%M:%S %p"),
        "red_alert_exposure": potential_default_value,
    }


def generate_pdf_report(df: pd.DataFrame, risk_sensitivity: float) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    summary = build_dashboard_summary(df, risk_sensitivity)
    stressed_df = summary["stressed_df"]
    top_10 = summary["top_10_df"].loc[
        :, ["borrower_id", "borrower_segment", "loan_amount", "emi_amount", "loan_health_score"]
    ]

    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(12, 12, "S", border=0, align="C", fill=True)
    pdf.set_xy(pdf.l_margin + 16, pdf.t_margin)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 6, "STELVORA Infrastructure", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 16)
    pdf.set_text_color(55, 65, 81)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Risk Audit Command Center", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 16)
    pdf.cell(0, 6, f"Report Date: {date.today().strftime('%d %B %Y')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 8, "STELVORA: QUARTERLY RISK AUDIT & NPA REDUCTION REPORT")
    pdf.ln(2)

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    col_width = page_width / 4
    headers = [
        "Total Portfolio",
        "Stressed Assets Detected",
        "Potential Default Value (Rs.)",
        "Predicted Recovery Savings",
    ]
    values = [
        format_pdf_compact_inr(float(summary["total_portfolio_value"])),
        f"{int(summary['stressed_assets_detected']):,}",
        format_pdf_compact_inr(float(summary["potential_default_value"])),
        format_pdf_compact_inr(float(summary["predicted_recovery_savings"])),
    ]

    pdf.set_fill_color(30, 58, 95)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for header in headers:
        pdf.multi_cell(
            col_width,
            8,
            header,
            border=1,
            align="C",
            fill=True,
            max_line_height=4,
            new_x="RIGHT",
            new_y="TOP",
        )
    pdf.ln(16)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "", 9)
    for value in values:
        pdf.multi_cell(
            col_width,
            8,
            value,
            border=1,
            align="C",
            max_line_height=4,
            new_x="RIGHT",
            new_y="TOP",
        )
    pdf.ln(14)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    analysis_text = (
        f"Stelvora's ML model identified {int(summary['stressed_assets_detected']):,} high-risk borrowers "
        f"before their next EMI date under the current dashboard filter: {summary['filter_label']}. "
        f"At a Risk Sensitivity of {risk_sensitivity:.2f}, this report treats borrowers with "
        f"Loan Health Score at or below {float(summary['score_cutoff']):.1f} as stressed."
    )
    pdf.multi_cell(0, 6, analysis_text)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top 10 Risk List", new_x="LMARGIN", new_y="NEXT")
    table_headers = ["Borrower ID", "Type", "Loan Amount", "EMI", "Health Score"]
    table_widths = [30, 38, 42, 35, 30]

    pdf.set_fill_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 9)
    for header, width in zip(table_headers, table_widths):
        pdf.cell(width, 8, header, border=1, align="C", fill=True)
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 9)
    for _, row in top_10.iterrows():
        pdf.cell(table_widths[0], 8, str(row["borrower_id"]), border=1)
        pdf.cell(table_widths[1], 8, str(row["borrower_segment"]), border=1)
        pdf.cell(table_widths[2], 8, format_pdf_currency(float(row["loan_amount"])), border=1, align="R")
        pdf.cell(table_widths[3], 8, format_pdf_currency(float(row["emi_amount"])), border=1, align="R")
        pdf.cell(table_widths[4], 8, f"{float(row['loan_health_score']):.2f}", border=1, align="R")
        pdf.ln(8)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(
        0,
        5,
        (
            f"This report matches the Stelvora Dashboard state at {summary['timestamp']} "
            f"with a Risk Sensitivity of {risk_sensitivity:.2f}."
        ),
    )

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


@st.cache_data(show_spinner="Preparing executive report...")
def build_executive_report_bytes(df: pd.DataFrame, risk_sensitivity: float) -> bytes:
    return generate_pdf_report(df, risk_sensitivity)


def render_sidebar_branding() -> None:
    brand_columns = st.sidebar.columns((0.34, 1))
    with brand_columns[0]:
        if FAVICON_PATH.exists():
            st.image(str(FAVICON_PATH), width=44)
        else:
            st.markdown(
                "<div style='width:44px;height:44px;border-radius:12px;background:#0f172a;"
                "border:1px solid rgba(148,163,184,0.24);display:flex;align-items:center;"
                "justify-content:center;color:#f8fafc;font-weight:800;font-size:1.2rem;'>S</div>",
                unsafe_allow_html=True,
            )
    with brand_columns[1]:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div>
                    <div class="sidebar-brand-title">STELVORA</div>
                    <div class="sidebar-brand-subtitle">Risk Command Center</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")


def render_metric_card(
    label: str,
    value: str,
    footnote: str,
    icon: str,
    accent_color: str,
    progress_pct: float,
    target_value: float,
    decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    pulse: bool = False,
) -> None:
    progress_pct = max(0.0, min(float(progress_pct), 100.0))
    card_id = (
        label.lower()
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )
    glow = (
        "box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.28), "
        f"0 0 26px {accent_color}44; animation: metricPulse 2.4s ease-in-out infinite;"
        if pulse
        else "box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.28);"
    )
    components.html(
        f"""
        <style>
            @keyframes metricPulse {{
                0% {{ box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.28), 0 0 12px {accent_color}22; }}
                50% {{ box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.28), 0 0 28px {accent_color}66; }}
                100% {{ box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 30px rgba(0,0,0,0.28), 0 0 12px {accent_color}22; }}
            }}
        </style>
        <div class="metric-card" style="background:#232426;border:1px solid rgba(255,255,255,0.07);border-radius:15px;padding:1.15rem 1.15rem 1rem 1.15rem;min-height:156px;{glow}">
            <div class="metric-top" style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.9rem;">
                <div class="metric-icon" style="color:{accent_color};font-size:0.95rem;">{icon}</div>
                <div class="metric-label" style="color:{TEXT_MUTED};font-size:0.96rem;">{label}</div>
            </div>
            <div id="metric-{card_id}" class="metric-value" style="color:{TEXT_PRIMARY};font-size:2.28rem;font-weight:700;margin-bottom:0.75rem;">{value}</div>
            <div class="metric-progress-track" style="width:100%;height:4px;border-radius:999px;background:rgba(255,255,255,0.06);overflow:hidden;margin:0.05rem 0 0.72rem 0;">
                <div class="metric-progress-fill" style="width:{progress_pct:.1f}%;height:100%;border-radius:999px;background:{accent_color};"></div>
            </div>
            <div class="metric-footnote" style="color:{TEXT_MUTED};font-size:0.9rem;">{footnote}</div>
        </div>
        <script>
            const target = {target_value:.6f};
            const decimals = {decimals};
            const prefix = {prefix!r};
            const suffix = {suffix!r};
            const el = document.getElementById("metric-{card_id}");
            const formatter = new Intl.NumberFormat('en-IN', {{
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            }});
            const duration = 1200;
            const start = performance.now();

            function step(now) {{
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const value = target * eased;
                el.textContent = prefix + formatter.format(value) + suffix;
                if (progress < 1) {{
                    requestAnimationFrame(step);
                }}
            }}
            requestAnimationFrame(step);
        </script>
        """,
        height=170,
    )


def render_top_header() -> None:
    today_label = date.today().strftime("%A, %d %B %Y")
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <div class="header-eyebrow">{today_label}</div>
                <div class="hero-title">Stelvora Dashboard</div>
                <div class="hero-subtitle">Loan health intelligence for Indian NBFC portfolios</div>
            </div>
            <div class="header-actions">
                <div class="header-pill">{today_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scanning_indicator() -> None:
    if st_lottie:
        lottie_payload = {
            "v": "5.7.4",
            "fr": 30,
            "ip": 0,
            "op": 90,
            "w": 120,
            "h": 120,
            "nm": "scan",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "ring",
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 90, "s": [360]}]},
                        "p": {"a": 0, "k": [60, 60, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {"a": 1, "k": [{"t": 0, "s": [80, 80, 100]}, {"t": 45, "s": [100, 100, 100]}, {"t": 90, "s": [80, 80, 100]}]},
                    },
                    "shapes": [
                        {"ty": "el", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [70, 70]}, "nm": "ellipse"},
                        {"ty": "st", "c": {"a": 0, "k": [0.56, 0.49, 0.97, 1]}, "o": {"a": 0, "k": 100}, "w": {"a": 0, "k": 6}},
                        {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
                    ],
                    "ip": 0,
                    "op": 90,
                    "st": 0,
                    "bm": 0,
                }
            ],
        }
        with st.container():
            st_lottie(lottie_payload, height=58, width=58, key="stelvora_scan")
    else:
        st.markdown(
            """
            <div class="scan-indicator">
                <span class="scan-dot"></span>
                <span class="scan-text">Scanning live portfolio signals...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_section_card_header(title: str, chip_label: str = "This Week") -> None:
    st.markdown(
        f"""
        <div class="section-card-header">
            <div class="section-card-title">{title}</div>
            <div class="section-card-chip">{chip_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_pills() -> None:
    st.markdown(
        f"""
        <div class="risk-pill-row">
            <div class="risk-pill"><span class="risk-pill-dot" style="background:{ACCENT_GREEN};"></span>Healthy &lt; 47%</div>
            <div class="risk-pill"><span class="risk-pill-dot" style="background:{ACCENT_PURPLE};"></span>Watchlist 47% - 65%</div>
            <div class="risk-pill"><span class="risk-pill-dot" style="background:{ACCENT_ORANGE};"></span>Critical &gt; 65%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_risk_bucket_badge(value: str) -> str:
    styles = {
        "Red": ("#ef4444", "#ffffff"),
        "Orange": ("#f97316", "#ffffff"),
        "Yellow": ("#eab308", "#111827"),
        "Green": ("#22c55e", "#ffffff"),
    }
    background, text = styles.get(value, ("rgba(148, 163, 184, 0.25)", "#f8fafc"))
    return (
        f"background-color: {background}; color: {text}; font-weight: 700; "
        "text-align: center; border-radius: 999px; padding: 0.18rem 0.55rem;"
    )


def build_watchlist_styler(watchlist: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        watchlist.style.format(
            {
                "loan_amount": "{:,.0f}",
                "emi_amount": "{:,.2f}",
                "loan_health_score": "{:.2f}",
            }
        )
        .map(style_risk_bucket_badge, subset=["risk_bucket"])
        .set_properties(
            subset=["borrower_id", "borrower_type", "loan_amount", "emi_amount", "loan_health_score"],
            **{"color": "#f8fafc"},
        )
    )


def build_risk_mix_donut_chart(df: pd.DataFrame) -> go.Figure:
    donut_df = pd.DataFrame(
        {
            "bucket": ["Green", "Orange", "Red"],
            "count": [
                int(df["risk_bucket"].isin(["Green", "Yellow"]).sum()),
                int(df["risk_bucket"].eq("Orange").sum()),
                int(df["risk_bucket"].eq("Red").sum()),
            ],
        }
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=donut_df["bucket"],
                values=donut_df["count"],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker=dict(colors=["#10B981", "#F59E0B", "#EF4444"], line=dict(color=CARD_BG, width=4)),
                textinfo="none",
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>{int(df['risk_bucket'].isin(['Orange', 'Red']).sum()):,}</b><br><span style='font-size:12px;color:#9a9aa0'>Stressed</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        align="center",
        font=dict(color=TEXT_PRIMARY, size=28),
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.0,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MUTED),
        ),
    )
    return fig


def build_watchlist_html_table(watchlist: pd.DataFrame) -> str:
    badge_styles = {
        "Red": ("rgba(255, 107, 107, 0.18)", "#ffd4d4"),
        "Orange": ("rgba(245, 177, 60, 0.18)", "#fbe4b0"),
        "Yellow": ("rgba(241, 211, 107, 0.2)", "#fff0b5"),
        "Green": ("rgba(103, 209, 122, 0.18)", "#d8ffe0"),
    }

    watchlist_styles = f"""
    <style>
        body {{
            margin: 0;
            background: transparent;
            color: {TEXT_PRIMARY};
            font-family: "Inter", "Roboto", "Segoe UI", sans-serif;
        }}

        .watchlist-shell {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
            overflow: hidden;
        }}

        .watchlist-scroll {{
            overflow-x: auto;
        }}

        .watchlist-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .watchlist-table thead th {{
            text-align: left;
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            font-weight: 600;
            padding: 0.95rem 1rem;
            border-bottom: 1px solid {CARD_BORDER};
            background: #2a2a2c;
            white-space: nowrap;
        }}

        .watchlist-table tbody td {{
            padding: 0.95rem 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            color: {TEXT_PRIMARY};
            font-size: 0.9rem;
            vertical-align: middle;
            white-space: nowrap;
        }}

        .watchlist-table tbody tr:hover {{
            background: rgba(255,255,255,0.02);
        }}

        .watchlist-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 74px;
            padding: 0.3rem 0.62rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
        }}

        .watchlist-prob {{
            min-width: 180px;
        }}

        .watchlist-prob-label {{
            color: {TEXT_MUTED};
            font-size: 0.76rem;
            margin-bottom: 0.35rem;
        }}

        .watchlist-prob-track {{
            width: 100%;
            height: 8px;
            background: #1f1f21;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        }}

        .watchlist-prob-fill {{
            height: 100%;
            border-radius: 999px;
        }}

        .watchlist-action {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.45rem 0.8rem;
            border-radius: 10px;
            font-size: 0.82rem;
            font-weight: 700;
            color: #111318;
            background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_ORANGE});
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
        }}
    </style>
    """

    rows = []
    for _, row in watchlist.iterrows():
        risk_bucket = str(row["risk_bucket"])
        badge_bg, badge_fg = badge_styles.get(risk_bucket, ("rgba(255,255,255,0.12)", TEXT_PRIMARY))
        risk_probability = max(0.0, min(100.0, 100.0 - float(row["loan_health_score"])))
        progress_color = {
            "Red": ACCENT_RED,
            "Orange": ACCENT_ORANGE,
            "Yellow": "#f1d36b",
            "Green": ACCENT_GREEN,
        }.get(risk_bucket, ACCENT_PURPLE)

        rows.append(
            f"""
            <tr>
                <td>{escape(str(row['borrower_id']))}</td>
                <td>{escape(str(row['borrower_type']))}</td>
                <td>{format_currency(float(row['loan_amount']))}</td>
                <td>{format_currency(float(row['emi_amount']))}</td>
                <td>{float(row['loan_health_score']):.2f}</td>
                <td><span class="watchlist-badge" style="background:{badge_bg};color:{badge_fg};">{escape(risk_bucket)}</span></td>
                <td class="watchlist-prob">
                    <div class="watchlist-prob-label">{risk_probability:.1f}% risk probability</div>
                    <div class="watchlist-prob-track">
                        <div class="watchlist-prob-fill" style="width:{risk_probability:.1f}%; background:{progress_color};"></div>
                    </div>
                </td>
                <td><span class="watchlist-action">Intervention</span></td>
            </tr>
            """
        )

    if not rows:
        rows.append(
            """
            <tr>
                <td colspan="8" style="text-align:center;color:#9a9aa0;">No borrowers match the current watchlist filter.</td>
            </tr>
            """
        )

    return (
        watchlist_styles
        + """
        <div class="watchlist-shell">
            <div class="watchlist-scroll">
                <table class="watchlist-table">
                    <thead>
                        <tr>
                            <th>Borrower ID</th>
                            <th>Type</th>
                            <th>Loan Amount</th>
                            <th>EMI</th>
                            <th>Health Score</th>
                            <th>Bucket</th>
                            <th>Risk Probability</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        + "".join(rows)
        + """
                    </tbody>
                </table>
            </div>
        </div>
        """
    )


def risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["risk_bucket"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )
    counts.columns = ["risk_bucket", "count"]

    fig = px.bar(
        counts,
        x="risk_bucket",
        y="count",
        color="risk_bucket",
        text="count",
        category_orders={"risk_bucket": RISK_ORDER},
        color_discrete_map=RISK_COLORS,
        template="plotly_dark",
    )
    fig.update_traces(showlegend=False, textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Risk Bucket",
        yaxis_title="Borrower Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 16, 30, 0.62)",
    )
    return fig


def borrower_type_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby(["borrower_segment", "portfolio_status"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        grouped,
        x="borrower_segment",
        y="count",
        color="portfolio_status",
        barmode="group",
        text="count",
        color_discrete_map={
            "Healthy": "#7B61FF",
            "Stressed": "#FF9F43",
        },
        category_orders={"portfolio_status": ["Healthy", "Stressed"]},
        labels={
            "borrower_segment": "Borrower Type",
            "count": "Borrower Count",
            "portfolio_status": "Status",
        },
        template="plotly_dark",
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        textfont=dict(color="#ffffff", size=13, family="Inter, Roboto, sans-serif"),
        cliponaxis=False,
        marker=dict(line=dict(color="rgba(255,255,255,0.08)", width=1)),
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            hovertemplate=f"<b>%{{x}}</b><br>{trace.name}: %{{y:,}}<extra></extra>"
        )
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        xaxis=dict(title="", tickfont=dict(color=TEXT_MUTED), showgrid=False),
        yaxis=dict(
            title="Borrower Count",
            tickfont=dict(color=TEXT_MUTED),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            automargin=True,
        ),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        hoverlabel=dict(bgcolor="#1f2023", bordercolor="rgba(255,255,255,0.08)", font=dict(color=TEXT_PRIMARY)),
    )
    return fig


def borrower_trend_chart(row: pd.Series) -> go.Figure:
    months = [f"Month {month}" for month in range(1, 7)]
    incomes = [row[f"month{month}_income"] for month in range(1, 7)]
    balances = [row[f"month{month}_balance"] for month in range(1, 7)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=months,
            y=incomes,
            name="Income",
            mode="lines+markers",
            line=dict(color="#7c80ff", width=3),
            marker=dict(size=9, color="#7c80ff", line=dict(color="#b6c1ff", width=1)),
            hovertemplate="<b>%{x}</b><br>Income: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=balances,
            name="Balance",
            mode="lines+markers",
            line=dict(color="#f5b13c", width=3),
            marker=dict(size=9, color="#f5b13c", line=dict(color="#ffdf99", width=1)),
            hovertemplate="<b>%{x}</b><br>Balance: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis_title="Month",
        yaxis_title="Amount",
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1f2023", bordercolor="rgba(255,255,255,0.08)", font=dict(color=TEXT_PRIMARY)),
    )
    return fig


def risk_bucket_from_score(score: float) -> str:
    if score > 75:
        return "Green"
    if score >= 50:
        return "Yellow"
    if score >= 30:
        return "Orange"
    return "Red"


def split_observed_window(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    midpoint = max(1, int(np.ceil(len(series) / 2)))
    first_half = series.iloc[:midpoint]
    second_half = series.iloc[midpoint:]
    if second_half.empty:
        second_half = first_half
    return first_half, second_half


def calculate_monthly_health_timeline(row: pd.Series) -> pd.DataFrame:
    income_values = [float(row[f"month{month}_income"]) for month in range(1, 7)]
    balance_values = [float(row[f"month{month}_balance"]) for month in range(1, 7)]
    payment_values = [bool(row[f"month{month}_paid"]) for month in range(1, 7)]

    records = []
    for month in range(1, 7):
        income_series = pd.Series(income_values[:month], dtype=float)
        balance_series = pd.Series(balance_values[:month], dtype=float)
        payment_series = pd.Series(payment_values[:month], dtype=bool)

        first_income, last_income = split_observed_window(income_series)
        first_balance, last_balance = split_observed_window(balance_series)

        income_trend_ratio = last_income.mean() / max(first_income.mean(), 1)
        balance_trend_ratio = last_balance.mean() / max(first_balance.mean(), 1)
        payment_rate = float(payment_series.mean()) if not payment_series.empty else 0.0
        projected_payment_count = int(np.floor(payment_rate * 6 + 1e-9))
        expense_pressure_ratio = balance_series.min() / max(balance_series.mean(), 1)
        income_volatility = income_series.std(ddof=0) / max(income_series.mean(), 1)

        income_trend_score = float(score_positive_trend(pd.Series([income_trend_ratio])).iloc[0])
        balance_trend_score = float(score_positive_trend(pd.Series([balance_trend_ratio])).iloc[0])
        payment_consistency_score = float(
            score_payment_consistency(pd.Series([projected_payment_count])).iloc[0]
        )
        expense_pressure_score = float(
            score_expense_pressure(pd.Series([expense_pressure_ratio])).iloc[0]
        )
        income_volatility_score = float(
            score_income_volatility(pd.Series([income_volatility])).iloc[0]
        )
        loan_health_score = round(
            (income_trend_score * 0.30)
            + (balance_trend_score * 0.25)
            + (payment_consistency_score * 0.20)
            + (expense_pressure_score * 0.15)
            + (income_volatility_score * 0.10),
            2,
        )

        records.append(
            {
                "month_number": month,
                "month_label": f"Month {month}",
                "income_trend_score": income_trend_score,
                "balance_trend_score": balance_trend_score,
                "payment_consistency_score": payment_consistency_score,
                "expense_pressure_score": expense_pressure_score,
                "income_volatility_score": income_volatility_score,
                "loan_health_score": loan_health_score,
                "risk_bucket": risk_bucket_from_score(loan_health_score),
            }
        )

    return pd.DataFrame(records)


def build_score_breakdown_data(row: pd.Series) -> pd.DataFrame:
    income_month1 = float(row["month1_income"])
    income_current = float(row["month6_income"])
    balance_month1 = float(row["month1_balance"])
    balance_current = float(row["month6_balance"])
    payment_count = int(sum(bool(row[f"month{month}_paid"]) for month in range(1, 7)))
    balances = pd.Series([float(row[f"month{month}_balance"]) for month in range(1, 7)], dtype=float)
    incomes = pd.Series([float(row[f"month{month}_income"]) for month in range(1, 7)], dtype=float)

    income_drop_pct = max(0.0, (1 - (income_current / max(income_month1, 1))) * 100)
    balance_drop_pct = max(0.0, (1 - (balance_current / max(balance_month1, 1))) * 100)
    expense_pressure_pct = (balances.min() / max(balances.mean(), 1)) * 100
    income_volatility_pct = (incomes.std(ddof=0) / max(incomes.mean(), 1)) * 100

    return pd.DataFrame(
        {
            "metric": [
                "Income Trend Score",
                "Balance Trend Score",
                "Payment Consistency Score",
                "Expense Pressure Score",
                "Income Volatility Score",
            ],
            "score": [
                float(row["income_trend_score"]),
                float(row["balance_trend_score"]),
                float(row["payment_consistency_score"]),
                float(row["expense_pressure_score"]),
                float(row["income_volatility_score"]),
            ],
            "explanation": [
                f"Income dropped {income_drop_pct:.1f}% from Month 1 to current month",
                f"Account balance declined {balance_drop_pct:.1f}% over the monitored period",
                f"Borrower made {payment_count} out of 6 EMI payments on time",
                f"Minimum balance dropped to {expense_pressure_pct:.1f}% of average balance",
                (
                    f"Income fluctuation of {income_volatility_pct:.1f}% month to month "
                    "— above 30% indicates instability"
                ),
            ],
        }
    )


def build_prediction_timeline_chart(
    timeline_df: pd.DataFrame,
    stress_month: int | None,
    default_month: int | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline_df["month_number"],
            y=timeline_df["loan_health_score"],
            mode="lines+markers",
            name="Loan Health Score",
            line=dict(color=ACCENT_PURPLE, width=3),
            marker=dict(size=10, color="#8f7cf7", line=dict(color="#d4d0ff", width=1)),
            hovertemplate="<b>Month %{x}</b><br>Loan Health Score: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=65,
        line_dash="dash",
        line_color=ACCENT_RED,
        annotation_text="Early Warning Threshold",
        annotation_position="top left",
    )

    if stress_month is not None:
        fig.add_vrect(
            x0=stress_month - 0.08,
            x1=stress_month + 0.08,
            fillcolor="rgba(245, 177, 60, 0.14)",
            line_width=0,
            layer="below",
        )
        fig.add_vline(
            x=stress_month,
            line_dash="dash",
            line_color=ACCENT_ORANGE,
            annotation_text="Stelvora detects stress here",
            annotation_position="top right",
        )

    if default_month is not None:
        default_score = float(
            timeline_df.loc[timeline_df["month_number"].eq(default_month), "loan_health_score"].iloc[0]
        )
        fig.add_trace(
            go.Scatter(
                x=[default_month],
                y=[default_score],
                mode="markers",
                name="Default Pulse",
                marker=dict(size=26, color="rgba(255, 107, 107, 0.14)", line=dict(color="rgba(255,107,107,0.22)", width=2)),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[default_month],
                y=[default_score],
                mode="markers",
                name="Default Month",
                marker=dict(symbol="x", size=16, color=ACCENT_RED, line=dict(width=2)),
                hovertemplate="<b>Month %{x}</b><br>Default recorded<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5, 6],
            ticktext=[f"Month {month}" for month in range(1, 7)],
        ),
        yaxis=dict(title="Loan Health Score", range=[0, 100]),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1f2023", bordercolor="rgba(255,255,255,0.08)", font=dict(color=TEXT_PRIMARY)),
    )
    return fig


def detect_early_warning_month(timeline_df: pd.DataFrame) -> int | None:
    scores = timeline_df["loan_health_score"].tolist()
    months = timeline_df["month_number"].tolist()
    if not scores:
        return None

    baseline_score = scores[0]

    for index in range(1, len(scores)):
        score_drop_from_month1 = baseline_score - scores[index]
        if score_drop_from_month1 > 15:
            return int(months[index])
    return None


def format_detection_gap(months_early: int) -> str:
    if months_early == 1:
        return "1 month early"
    return f"{months_early} months early"


def build_score_breakdown_chart(row: pd.Series) -> go.Figure:
    breakdown = build_score_breakdown_data(row)
    breakdown["color"] = breakdown["score"].apply(
        lambda score: "#22c55e" if score > 70 else "#f97316" if score >= 30 else "#ef4444"
    )

    fig = go.Figure(
        go.Bar(
            x=breakdown["score"],
            y=breakdown["metric"],
            orientation="h",
            marker_color=breakdown["color"],
            text=breakdown["score"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    for _, breakdown_row in breakdown.iterrows():
        fig.add_annotation(
            x=0.02,
            y=breakdown_row["metric"],
            xref="paper",
            yref="y",
            text=breakdown_row["explanation"],
            showarrow=False,
            xanchor="left",
            yanchor="top",
            yshift=-18,
            font=dict(size=11, color="#cbd5e1"),
            align="left",
        )
    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Score", range=[0, 100]),
        yaxis=dict(title="", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 16, 30, 0.62)",
        showlegend=False,
        bargap=0.52,
    )
    return fig


def main() -> None:
    apply_dark_theme()

    if not DATA_PATH.exists() or not SCORES_PATH.exists():
        st.error("Missing loan_data.csv or loan_scores.csv in the current folder.")
        st.stop()

    portfolio = load_portfolio()
    render_sidebar_branding()
    st.sidebar.markdown("### Controls")
    risk_sensitivity = st.sidebar.slider(
        "Risk Sensitivity",
        min_value=0.30,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Higher sensitivity widens the executive intervention list for the PDF report.",
    )
    available_segments = sorted(portfolio["borrower_segment"].dropna().unique().tolist())
    selected_segments = st.sidebar.multiselect(
        "Borrower Filter",
        options=available_segments,
        default=available_segments,
        help="The dashboard and PDF report will use only the borrower segments selected here.",
    )
    visible_portfolio = filter_portfolio_by_segment(portfolio, selected_segments)
    score_cutoff = get_intervention_cutoff(risk_sensitivity)
    if visible_portfolio.empty:
        st.warning("No borrowers match the current filter selection.")
        st.stop()

    visible_portfolio = visible_portfolio.copy()
    visible_portfolio["portfolio_status"] = visible_portfolio["loan_health_score"].apply(
        lambda score: "Stressed" if score <= score_cutoff else "Healthy"
    )

    dashboard_summary = build_dashboard_summary(visible_portfolio, risk_sensitivity)
    intervention_watchlist = dashboard_summary["stressed_df"]
    executive_report_bytes = build_executive_report_bytes(visible_portfolio, risk_sensitivity)
    st.sidebar.caption(
        f"Executive report matches the current filter and flags Loan Health Score at or below {score_cutoff:.1f}."
    )
    st.sidebar.download_button(
        "📄 Download Executive Report",
        data=executive_report_bytes,
        file_name="Stelvora_Quarterly_Risk_Audit_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    total_active_loans = int(dashboard_summary["total_loans"])
    stressed_loans = int(dashboard_summary["stressed_assets_detected"])
    portfolio_health_score = float(dashboard_summary["portfolio_health_score"])
    red_alert_exposure = float(dashboard_summary["red_alert_exposure"])
    total_portfolio_value = float(dashboard_summary["total_portfolio_value"])
    stressed_progress = (stressed_loans / total_active_loans * 100) if total_active_loans else 0.0
    red_alert_progress = (
        (red_alert_exposure / total_portfolio_value) * 100 if total_portfolio_value else 0.0
    )
    red_alert_exposure_cr = red_alert_exposure / 1_00_00_000 if red_alert_exposure else 0.0

    render_top_header()
    render_scanning_indicator()

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Total Active Loans",
            f"{total_active_loans:,}",
            "Borrowers currently tracked in the portfolio",
            "🏦",
            ACCENT_GREEN,
            100.0,
            target_value=float(total_active_loans),
            decimals=0,
        )
    with metric_columns[1]:
        render_metric_card(
            "Predicted Stressed Loans",
            f"{stressed_loans:,}",
            f"Borrowers at or below the active cutoff of {score_cutoff:.1f}",
            "⚠️",
            ACCENT_ORANGE,
            stressed_progress,
            target_value=float(stressed_loans),
            decimals=0,
        )
    with metric_columns[2]:
        render_metric_card(
            "Portfolio Health Score",
            f"{portfolio_health_score:.1f}",
            "Average borrower score across the full portfolio",
            "📊",
            ACCENT_PURPLE,
            portfolio_health_score,
            target_value=float(portfolio_health_score),
            decimals=1,
        )
    with metric_columns[3]:
        render_metric_card(
            "Red Alert Exposure (Rs Cr)",
            f"₹{red_alert_exposure_cr:.2f} Cr",
            "Total loan amount for borrowers visible in the active intervention cohort",
            "🚨",
            ACCENT_RED,
            red_alert_progress,
            target_value=float(red_alert_exposure_cr),
            decimals=2,
            prefix="₹",
            suffix=" Cr",
            pulse=True,
        )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        render_section_card_header("Healthy vs. Stressed", "By Borrower Type")
        render_risk_pills()
        st.plotly_chart(
            borrower_type_breakdown_chart(visible_portfolio),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with chart_columns[1]:
        render_section_card_header("Risk Mix", "Current Portfolio")
        st.plotly_chart(
            build_risk_mix_donut_chart(visible_portfolio),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.markdown('<div class="section-label">Priority Watchlist</div>', unsafe_allow_html=True)
    watchlist_filter = st.selectbox(
        "Show borrowers in risk bucket",
        ["All Stressed", "Red Only", "Orange Only", "Yellow Only"],
        index=0,
    )
    if watchlist_filter == "Red Only":
        watchlist_source = visible_portfolio.loc[visible_portfolio["risk_bucket"].eq("Red")]
    elif watchlist_filter == "Orange Only":
        watchlist_source = visible_portfolio.loc[visible_portfolio["risk_bucket"].eq("Orange")]
    elif watchlist_filter == "Yellow Only":
        watchlist_source = visible_portfolio.loc[visible_portfolio["risk_bucket"].eq("Yellow")]
    else:
        watchlist_source = visible_portfolio.loc[
            visible_portfolio["risk_bucket"].isin(["Orange", "Red"])
        ]

    watchlist = (
        watchlist_source
        .loc[
            :,
            [
                "borrower_id",
                "borrower_segment",
                "loan_amount",
                "emi_amount",
                "loan_health_score",
                "risk_bucket",
            ],
        ]
        .rename(columns={"borrower_segment": "borrower_type"})
        .sort_values("loan_health_score", ascending=True)
        .reset_index(drop=True)
        )
    watchlist_control_cols = st.columns((1.15, 1.45, 0.7))
    with watchlist_control_cols[0]:
        st.markdown(f"Showing **{len(watchlist):,}** borrowers")
    with watchlist_control_cols[1]:
        watchlist_borrower_options = watchlist["borrower_id"].tolist()
        intervention_target = st.selectbox(
            "Trigger intervention for borrower",
            watchlist_borrower_options if watchlist_borrower_options else ["No borrowers available"],
            disabled=not bool(watchlist_borrower_options),
            key="watchlist_intervention_target",
        )
    with watchlist_control_cols[2]:
        st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
        if st.button(
            "Intervention",
            key="watchlist_intervention_button",
            disabled=not bool(watchlist_borrower_options),
            use_container_width=True,
        ):
            st.toast(
                (
                    f"✅ WhatsApp intervention triggered for {intervention_target} — "
                    "EMI split option sent in regional language. Stage 1 of 6 escalation ladder activated."
                )
            )
    watchlist_html = build_watchlist_html_table(watchlist)
    components.html(watchlist_html, height=500, scrolling=True)

    st.markdown('<div class="section-label">Executive Report</div>', unsafe_allow_html=True)
    report_columns = st.columns((1.4, 1))
    predicted_recovery_savings = (
        float(dashboard_summary["predicted_recovery_savings"])
    )
    with report_columns[0]:
        st.markdown(
            (
                f"Stelvora's executive report currently flags **{stressed_loans:,}** borrowers "
                f"for immediate outreach at a risk sensitivity of **{risk_sensitivity:.2f}** within **{dashboard_summary['filter_label']}**, "
                f"representing **{format_currency(float(dashboard_summary['potential_default_value']))}** in potential default value "
                f"and **{format_currency(predicted_recovery_savings)}** in predicted recovery savings."
            )
        )
    with report_columns[1]:
        st.info("Use the sidebar button to download the print-ready PDF report.")

    st.markdown('<div class="section-label">Borrower Deep Dive</div>', unsafe_allow_html=True)
    borrower_ids = visible_portfolio["borrower_id"].tolist()
    selected_borrower_id = st.selectbox("Select borrower_id", borrower_ids, index=0)
    borrower = visible_portfolio.loc[visible_portfolio["borrower_id"].eq(selected_borrower_id)].iloc[0]
    timeline_df = calculate_monthly_health_timeline(borrower)

    stress_month = detect_early_warning_month(timeline_df)
    missed_months = [
        month for month in range(1, 7) if not bool(borrower[f"month{month}_paid"])
    ]
    default_month = missed_months[0] if missed_months else None
    months_early = 0
    if stress_month is not None and default_month is not None:
        months_early = max(0, default_month - stress_month)

    deep_dive_columns = st.columns((1.6, 1))
    with deep_dive_columns[0]:
        st.plotly_chart(
            borrower_trend_chart(borrower),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with deep_dive_columns[1]:
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="section-label" style="margin-top: 0;">Borrower Profile</div>
                <div class="profile-item-label">Borrower Type</div>
                <div class="profile-item-value">{borrower['borrower_segment']}</div>
                <div class="profile-item-label">Loan Amount</div>
                <div class="profile-item-value">{format_currency(float(borrower['loan_amount']))}</div>
                <div class="profile-item-label">EMI Amount</div>
                <div class="profile-item-value">{format_currency(float(borrower['emi_amount']))}</div>
                <div class="profile-item-label">Loan Health Score</div>
                <div class="profile-item-value">{float(borrower['loan_health_score']):.2f}</div>
                <div class="profile-item-label">Risk Bucket</div>
                <div class="profile-item-value">{borrower['risk_bucket']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if borrower["risk_bucket"] in ["Orange", "Red"]:
            st.error(
                f"Risk alert: {borrower['borrower_id']} is in the {borrower['risk_bucket']} bucket and requires attention."
            )
        else:
            st.success(
                f"{borrower['borrower_id']} is currently in the {borrower['risk_bucket']} bucket."
            )

    st.markdown('<div class="section-label">Prediction Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_prediction_timeline_chart(timeline_df, stress_month, default_month),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )
    if default_month is not None:
        if stress_month is None or months_early <= 0:
            badge_text = "⚠️ Sudden default detected — limited intervention window"
        elif months_early == 1:
            badge_text = "⚠️ Stelvora detected risk 1 month before EMI was missed"
        else:
            badge_text = (
                f"⚠️ Stelvora detected risk {months_early} months before EMI was missed "
                "— intervention window available"
            )
        st.markdown(f'<div class="timeline-badge">{badge_text}</div>', unsafe_allow_html=True)
    else:
        st.info("No EMI miss was recorded for this borrower in the observed six-month window.")
    st.info(
        "Traditional systems detect risk only after EMI is missed. Stelvora detects risk before default using behavioral signals — giving lenders 2 to 3 months to act."
    )

    st.markdown('<div class="section-label">Score Breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_score_breakdown_chart(borrower),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown('<div class="section-label">Before vs After Stelvora</div>', unsafe_allow_html=True)
    comparison_columns = st.columns(2)
    left_outcome = (
        "EMI missed, collections begin, NPA declared"
        if default_month is not None
        else "No EMI missed in observed period, but no proactive risk workflow existed"
    )
    left_principal = float(borrower["loan_amount"]) if default_month is not None else 0.0
    right_detection = (
        format_detection_gap(months_early)
        if default_month is not None and stress_month is not None
        else format_detection_gap(0)
    )

    with comparison_columns[0]:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-title">Without Stelvora</div>
                <div class="comparison-item"><b>Risk detected:</b> Never</div>
                <div class="comparison-item"><b>Action taken:</b> None</div>
                <div class="comparison-item"><b>Outcome:</b> {left_outcome}</div>
                <div class="comparison-item"><b>Principal lost:</b> {format_currency(left_principal)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comparison_columns[1]:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-title">With Stelvora</div>
                <div class="comparison-item"><b>Risk detected:</b> {right_detection}</div>
                <div class="comparison-item"><b>Action taken:</b> Automated WhatsApp intervention sent</div>
                <div class="comparison-item"><b>Outcome:</b> EMI restructured, loan stays healthy</div>
                <div class="comparison-item"><b>Principal saved:</b> {format_currency(float(borrower['loan_amount']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
