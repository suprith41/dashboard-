from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("loan_data.csv")
OUTPUT_PATH = Path(__file__).with_name("loan_scores.csv")
OUTPUT_COLUMNS = [
    "borrower_id",
    "income_trend_score",
    "balance_trend_score",
    "payment_consistency_score",
    "expense_pressure_score",
    "income_volatility_score",
    "loan_health_score",
    "risk_bucket",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def score_positive_trend(ratio: pd.Series) -> pd.Series:
    ratio = ratio.clip(lower=0)
    score = pd.Series(index=ratio.index, dtype=float)

    high_mask = ratio > 1.0
    mid_mask = (ratio >= 0.7) & (ratio <= 1.0)
    low_mask = (ratio >= 0.4) & (ratio < 0.7)
    weak_mask = ratio < 0.4

    score.loc[high_mask] = 100.0
    score.loc[mid_mask] = 50 + ((ratio.loc[mid_mask] - 0.7) / 0.3) * 50
    score.loc[low_mask] = 20 + ((ratio.loc[low_mask] - 0.4) / 0.3) * 30
    score.loc[weak_mask] = (ratio.loc[weak_mask] / 0.4) * 20

    return score.clip(0, 100).round(2)


def score_payment_consistency(payment_count: pd.Series) -> pd.Series:
    score_map = {
        6: 100,
        5: 83,
        4: 67,
        3: 50,
        2: 33,
        1: 17,
        0: 0,
    }
    return payment_count.map(score_map).astype(float)


def score_expense_pressure(ratio: pd.Series) -> pd.Series:
    ratio = ratio.clip(lower=0)
    score = pd.Series(index=ratio.index, dtype=float)

    strong_mask = ratio > 0.5
    medium_mask = (ratio >= 0.2) & (ratio <= 0.5)
    weak_mask = (ratio >= 0.1) & (ratio < 0.2)
    critical_mask = ratio < 0.1

    score.loc[strong_mask] = 100.0
    score.loc[medium_mask] = 40 + ((ratio.loc[medium_mask] - 0.2) / 0.3) * 60
    score.loc[weak_mask] = 10 + ((ratio.loc[weak_mask] - 0.1) / 0.1) * 30
    score.loc[critical_mask] = (ratio.loc[critical_mask] / 0.1) * 10

    return score.clip(0, 100).round(2)


def score_income_volatility(volatility: pd.Series) -> pd.Series:
    volatility = volatility.clip(lower=0)
    score = pd.Series(index=volatility.index, dtype=float)

    stable_mask = volatility < 0.1
    moderate_mask = (volatility >= 0.1) & (volatility <= 0.3)
    elevated_mask = (volatility > 0.3) & (volatility <= 0.6)
    high_mask = volatility > 0.6

    score.loc[stable_mask] = 100.0
    score.loc[moderate_mask] = 100 - ((volatility.loc[moderate_mask] - 0.1) / 0.2) * 40
    score.loc[elevated_mask] = 60 - ((volatility.loc[elevated_mask] - 0.3) / 0.3) * 40
    score.loc[high_mask] = 20 - ((volatility.loc[high_mask] - 0.6) / 0.6) * 20

    return score.clip(0, 100).round(2)


def assign_risk_bucket(score: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [score > 75, score >= 50, score >= 30],
            ["Green", "Yellow", "Orange"],
            default="Red",
        ),
        index=score.index,
    )


def build_loan_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    early_income_avg = df[[f"month{month}_income" for month in range(1, 4)]].mean(axis=1)
    late_income_avg = df[[f"month{month}_income" for month in range(4, 7)]].mean(axis=1)
    early_balance_avg = df[[f"month{month}_balance" for month in range(1, 4)]].mean(axis=1)
    late_balance_avg = df[[f"month{month}_balance" for month in range(4, 7)]].mean(axis=1)

    all_income_avg = df[[f"month{month}_income" for month in range(1, 7)]].mean(axis=1)
    all_balance_avg = df[[f"month{month}_balance" for month in range(1, 7)]].mean(axis=1)
    minimum_balance = df[[f"month{month}_balance" for month in range(1, 7)]].min(axis=1)
    payment_count = df[[f"month{month}_paid" for month in range(1, 7)]].sum(axis=1)
    income_volatility = (
        df[[f"month{month}_income" for month in range(1, 7)]].std(axis=1, ddof=0)
        / all_income_avg.clip(lower=1)
    )

    income_trend_ratio = late_income_avg / early_income_avg.clip(lower=1)
    balance_trend_ratio = late_balance_avg / early_balance_avg.clip(lower=1)
    expense_pressure_ratio = minimum_balance / all_balance_avg.clip(lower=1)

    scores = pd.DataFrame(index=df.index)
    scores["borrower_id"] = df["borrower_id"]
    scores["income_trend_score"] = score_positive_trend(income_trend_ratio)
    scores["balance_trend_score"] = score_positive_trend(balance_trend_ratio)
    scores["payment_consistency_score"] = score_payment_consistency(payment_count)
    scores["expense_pressure_score"] = score_expense_pressure(expense_pressure_ratio)
    scores["income_volatility_score"] = score_income_volatility(income_volatility)

    scores["loan_health_score"] = (
        (scores["income_trend_score"] * 0.30)
        + (scores["balance_trend_score"] * 0.25)
        + (scores["payment_consistency_score"] * 0.20)
        + (scores["expense_pressure_score"] * 0.15)
        + (scores["income_volatility_score"] * 0.10)
    ).round(2)
    scores["risk_bucket"] = assign_risk_bucket(scores["loan_health_score"])

    return scores[OUTPUT_COLUMNS]


def print_summary(scores: pd.DataFrame) -> None:
    average_score = scores["loan_health_score"].mean()
    min_score = scores["loan_health_score"].min()
    max_score = scores["loan_health_score"].max()
    median_score = scores["loan_health_score"].median()

    print(f"Average loan health score across all borrowers: {average_score:.2f}")
    print(
        "Score distribution — "
        f"min: {min_score:.2f}, max: {max_score:.2f}, median: {median_score:.2f}"
    )

    print("\nRisk bucket counts:")
    bucket_counts = scores["risk_bucket"].value_counts().reindex(
        ["Green", "Yellow", "Orange", "Red"],
        fill_value=0,
    )
    for bucket, count in bucket_counts.items():
        print(f"{bucket}: {count}")

    print("\nTop 5 most at risk borrowers:")
    top_risk = scores.sort_values(["loan_health_score", "borrower_id"]).head(5)
    print(
        top_risk[["borrower_id", "loan_health_score", "risk_bucket"]].to_string(index=False)
    )

    print("\nSample of 5 random borrowers:")
    sample = scores.sample(n=5, random_state=1)
    print(
        sample[
            [
                "borrower_id",
                "income_trend_score",
                "balance_trend_score",
                "payment_consistency_score",
                "expense_pressure_score",
                "income_volatility_score",
                "loan_health_score",
                "risk_bucket",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    df = load_dataset(DATA_PATH)
    scores = build_loan_health_scores(df)
    scores.to_csv(OUTPUT_PATH, index=False)
    print_summary(scores)


if __name__ == "__main__":
    main()
