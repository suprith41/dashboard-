from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("loan_data.csv")
OUTPUT_PATH = Path(__file__).with_name("shadow_pilot_report.csv")
PDF_OUTPUT_PATH = Path(__file__).with_name("Stelvora_Impact_Report.pdf")
PREDICTION_THRESHOLD = 0.50


def format_rupees(value: float) -> str:
    return f"Rs {value:,.0f}"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_early_period_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    avg_income_1_3 = df[[f"month{month}_income" for month in range(1, 4)]].mean(axis=1)
    avg_balance_1_3 = df[[f"month{month}_balance" for month in range(1, 4)]].mean(axis=1)

    features["loan_amount"] = df["loan_amount"]
    features["emi_amount"] = df["emi_amount"]
    features["avg_monthly_income"] = df["avg_monthly_income"]
    features["month1_income"] = df["month1_income"]
    features["month2_income"] = df["month2_income"]
    features["month3_income"] = df["month3_income"]
    features["month1_balance"] = df["month1_balance"]
    features["month2_balance"] = df["month2_balance"]
    features["month3_balance"] = df["month3_balance"]
    features["income_trend_early"] = df["month3_income"] / df["month1_income"].clip(lower=1)
    features["balance_trend_early"] = df["month3_balance"] / df["month1_balance"].clip(lower=1)
    features["income_change_pct_1_3"] = (
        (df["month3_income"] - df["month1_income"]) / df["month1_income"].clip(lower=1)
    )
    features["balance_change_pct_1_3"] = (
        (df["month3_balance"] - df["month1_balance"]) / df["month1_balance"].clip(lower=1)
    )
    features["income_volatility_1_3"] = (
        df[[f"month{month}_income" for month in range(1, 4)]].std(axis=1)
        / avg_income_1_3.clip(lower=1)
    )
    features["balance_volatility_1_3"] = (
        df[[f"month{month}_balance" for month in range(1, 4)]].std(axis=1)
        / avg_balance_1_3.clip(lower=1)
    )
    features["payment_consistency_1_3"] = (
        df[[f"month{month}_paid" for month in range(1, 4)]].astype(float).mean(axis=1)
    )
    features["emi_to_income_1_3"] = df["emi_amount"] / avg_income_1_3.clip(lower=1)
    features["balance_to_income_1_3"] = avg_balance_1_3 / avg_income_1_3.clip(lower=1)

    borrower_flags = pd.get_dummies(df["borrower_type"], prefix="borrower_type")
    return pd.concat([features, borrower_flags], axis=1)


def actual_late_period_outcome(df: pd.DataFrame) -> pd.Series:
    late_payments = df[[f"month{month}_paid" for month in range(4, 7)]]
    return (~late_payments.all(axis=1)).astype(int)


def build_shadow_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def assign_pilot_risk_bucket(default_probability: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                default_probability >= 0.80,
                default_probability >= 0.60,
                default_probability >= 0.35,
            ],
            ["Red", "Orange", "Yellow"],
            default="Green",
        ),
        index=default_probability.index,
    )


def run_shadow_pilot_analysis(
    data_path: Path = DATA_PATH,
    output_csv_path: Path | None = OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    df = load_dataset(data_path)
    early_features = build_early_period_features(df)
    actual_defaults = actual_late_period_outcome(df)

    model = build_shadow_model()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    default_probabilities = cross_val_predict(
        model,
        early_features,
        actual_defaults,
        cv=folds,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    predicted_defaults = (default_probabilities >= PREDICTION_THRESHOLD).astype(int)

    true_positives = int(((predicted_defaults == 1) & (actual_defaults == 1)).sum())
    false_negatives = int(((predicted_defaults == 0) & (actual_defaults == 1)).sum())
    false_positives = int(((predicted_defaults == 1) & (actual_defaults == 0)).sum())
    total_actual_defaults = int(actual_defaults.sum())

    precision = precision_score(actual_defaults, predicted_defaults, zero_division=0) * 100
    recall = recall_score(actual_defaults, predicted_defaults, zero_division=0) * 100
    matrix = confusion_matrix(actual_defaults, predicted_defaults)

    identified_amount_at_risk = float(
        df.loc[(predicted_defaults == 1) & (actual_defaults == 1), "loan_amount"].sum()
    )
    prevented_npas = int(round(true_positives * 0.70))
    estimated_principal_saved = identified_amount_at_risk * 0.70
    estimated_recovery_cost_saved = true_positives * 8000
    estimated_total_savings = estimated_recovery_cost_saved + estimated_principal_saved
    npa_reduction_rate = (
        (prevented_npas / total_actual_defaults) * 100 if total_actual_defaults else 0.0
    )

    report = df.copy()
    report["actual_default_late_period"] = actual_defaults.astype(bool)
    report["predicted_default"] = predicted_defaults.astype(bool)
    report["default_probability"] = np.round(default_probabilities, 4)
    report["pilot_health_score"] = np.round((1 - default_probabilities) * 100, 2)
    report["pilot_risk_bucket"] = assign_pilot_risk_bucket(pd.Series(default_probabilities))
    report["prediction_result"] = np.select(
        [
            report["predicted_default"] & report["actual_default_late_period"],
            report["predicted_default"] & ~report["actual_default_late_period"],
            ~report["predicted_default"] & report["actual_default_late_period"],
        ],
        ["Correctly Predicted Default", "False Alarm", "Missed Default"],
        default="Correctly Cleared",
    )

    if output_csv_path is not None:
        report.to_csv(output_csv_path, index=False)

    summary = {
        "total_loans_analysed": len(df),
        "total_actual_defaults": total_actual_defaults,
        "defaults_predicted_correctly": true_positives,
        "defaults_missed": false_negatives,
        "false_alarms": false_positives,
        "precision_percentage": precision,
        "recall_percentage": recall,
        "identified_amount_at_risk": identified_amount_at_risk,
        "prevented_npas": prevented_npas,
        "estimated_principal_saved": estimated_principal_saved,
        "estimated_recovery_cost_saved": estimated_recovery_cost_saved,
        "estimated_total_savings": estimated_total_savings,
        "npa_reduction_rate": npa_reduction_rate,
        "confusion_matrix": matrix,
    }
    return report, summary


def build_high_risk_table(report: pd.DataFrame) -> pd.DataFrame:
    high_risk = report.loc[
        report["pilot_risk_bucket"].eq("Red") & report["actual_default_late_period"],
        [
            "borrower_id",
            "borrower_type",
            "loan_amount",
            "emi_amount",
            "default_probability",
            "pilot_health_score",
        ],
    ].sort_values(["default_probability", "loan_amount"], ascending=[False, False])
    return high_risk.head(20).reset_index(drop=True)


def build_pdf_report(
    report: pd.DataFrame,
    summary: dict[str, float | int],
    output_path: Path = PDF_OUTPUT_PATH,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "reportlab is required to generate Stelvora_Impact_Report.pdf."
        ) from exc

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AuditTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "AuditSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
        spaceBefore=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
    )

    executive_rows = [
        ["Metric", "Value"],
        ["Total Loans Audited", f"{summary['total_loans_analysed']:,}"],
        ["Total NPAs in Historical Data", f"{summary['total_actual_defaults']:,}"],
        ["NPAs Predicted by Stelvora", f"{summary['defaults_predicted_correctly']:,}"],
        ["NPA Reduction Rate (%)", f"{summary['npa_reduction_rate']:.2f}%"],
    ]

    high_risk = build_high_risk_table(report)
    high_risk_rows = [
        [
            "Borrower ID",
            "Type",
            "Loan Amount",
            "EMI",
            "Default Probability",
            "Pilot Health Score",
        ]
    ]
    for _, row in high_risk.iterrows():
        high_risk_rows.append(
            [
                row["borrower_id"],
                row["borrower_type"],
                format_rupees(float(row["loan_amount"])),
                format_rupees(float(row["emi_amount"])),
                f"{float(row['default_probability']) * 100:.2f}%",
                f"{float(row['pilot_health_score']):.2f}",
            ]
        )

    financial_text = (
        "<b>Estimated Savings:</b> "
        f"<b>{format_rupees(float(summary['estimated_total_savings']))}</b><br/>"
        f"Recovery cost savings: <b>{format_rupees(float(summary['estimated_recovery_cost_saved']))}</b><br/>"
        f"Principal saved: <b>{format_rupees(float(summary['estimated_principal_saved']))}</b>"
    )
    intervention_text = (
        "These borrowers showed clear early-warning stress and could have been prioritized "
        "for the Stelvora Flexible EMI workflow. With proactive outreach, EMI restructuring, "
        "and assisted collections support, this cohort represented the strongest opportunity "
        "to reduce NPA formation before final delinquency."
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    story = [
        Paragraph("Stelvora Risk Audit", title_style),
        Paragraph(f"Executive impact review generated on {date.today().strftime('%B %d, %Y')}", subtitle_style),
        Spacer(1, 6),
        Paragraph("Executive Summary", section_style),
    ]

    executive_table = Table(executive_rows, colWidths=[95 * mm, 65 * mm])
    executive_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([executive_table, Spacer(1, 12)])
    story.extend(
        [
            Paragraph("Financial Impact", section_style),
            Paragraph(financial_text, body_style),
            Spacer(1, 12),
            Paragraph("High-Risk List", section_style),
        ]
    )

    high_risk_table = Table(
        high_risk_rows,
        colWidths=[22 * mm, 26 * mm, 28 * mm, 22 * mm, 34 * mm, 28 * mm],
        repeatRows=1,
    )
    high_risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([high_risk_table, Spacer(1, 12)])
    story.extend(
        [
            Paragraph("Intervention Strategy", section_style),
            Paragraph(intervention_text, body_style),
        ]
    )

    document.build(story)
    return output_path.read_bytes()


def generate_impact_report_assets(
    data_path: Path = DATA_PATH,
    output_csv_path: Path | None = OUTPUT_PATH,
    output_pdf_path: Path = PDF_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, float | int], bytes]:
    report, summary = run_shadow_pilot_analysis(data_path=data_path, output_csv_path=output_csv_path)
    pdf_bytes = build_pdf_report(report, summary, output_path=output_pdf_path)
    return report, summary, pdf_bytes


def main() -> None:
    report, summary = run_shadow_pilot_analysis()
    pdf_created = False
    try:
        build_pdf_report(report, summary)
        pdf_created = True
    except RuntimeError as exc:
        print(str(exc))

    print(f"Total loans analysed: {summary['total_loans_analysed']:,}")
    print(f"Total actual defaults: {summary['total_actual_defaults']:,}")
    print(f"Defaults Stelvora predicted correctly: {summary['defaults_predicted_correctly']:,}")
    print(f"Defaults Stelvora missed: {summary['defaults_missed']:,}")
    print(f"False alarms: {summary['false_alarms']:,}")
    print(f"Precision percentage: {summary['precision_percentage']:.2f}%")
    print(f"Recall percentage: {summary['recall_percentage']:.2f}%")
    print(
        "Total loan amount at risk that Stelvora identified in rupees: "
        f"{format_rupees(float(summary['identified_amount_at_risk']))}"
    )
    print(
        "Estimated principal saved assuming Stelvora prevented 70 percent of correctly predicted defaults: "
        f"{format_rupees(float(summary['estimated_principal_saved']))}"
    )
    print(
        "Estimated recovery cost saved at rupees 5000 per prevented NPA: "
        f"{format_rupees(float(summary['prevented_npas'] * 5000))}"
    )
    if pdf_created:
        print(f"Executive impact report created: {PDF_OUTPUT_PATH.name}")
    else:
        print("Executive impact report not created in this environment.")
    print("Confusion matrix:")
    print(summary["confusion_matrix"])
    print(
        f"Shadow pilot complete. Stelvora would have prevented {summary['prevented_npas']} NPAs saving Rs {float(summary['estimated_principal_saved']) / 1_00_00_000:.2f} crore."
    )


if __name__ == "__main__":
    main()
