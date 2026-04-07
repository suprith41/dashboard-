import numpy as np
import pandas as pd


NUM_ROWS = 10000
DEFAULT_COUNT = 1500
HEALTHY_COUNT = NUM_ROWS - DEFAULT_COUNT
OUTPUT_FILE = "loan_data.csv"
BORROWER_TYPES = ["salaried", "gig_worker", "farmer", "small_trader"]
STRESS_GROUPS = [
    {
        "name": "mild",
        "count": 400,
        "income_ratios": np.array([1.00, 0.88, 0.78, 0.68, 0.60, 0.55], dtype=float),
        "balance_ratios": np.array([1.00, 0.85, 0.72, 0.60, 0.50, 0.42], dtype=float),
        "payments": np.array([True, True, True, True, True, False], dtype=bool),
    },
    {
        "name": "moderate",
        "count": 500,
        "income_ratios": np.array([1.00, 0.78, 0.58, 0.42, 0.30, 0.20], dtype=float),
        "balance_ratios": np.array([1.00, 0.72, 0.52, 0.35, 0.22, 0.12], dtype=float),
        "payments": np.array([True, True, True, True, False, False], dtype=bool),
    },
    {
        "name": "severe",
        "count": 400,
        "income_ratios": np.array([1.00, 0.65, 0.45, 0.28, 0.18, 0.10], dtype=float),
        "balance_ratios": np.array([1.00, 0.58, 0.38, 0.22, 0.10, 0.05], dtype=float),
        "payments": np.array([True, True, True, False, False, False], dtype=bool),
    },
    {
        "name": "critical",
        "count": 200,
        "income_ratios": np.array([1.00, 0.55, 0.35, 0.20, 0.10, 0.05], dtype=float),
        "balance_ratios": np.array([1.00, 0.45, 0.25, 0.12, 0.05, 0.02], dtype=float),
        "payments": np.array([True, True, False, False, False, False], dtype=bool),
    },
]


def make_even_borrower_types(count: int) -> np.ndarray:
    repeated = np.repeat(BORROWER_TYPES, count // len(BORROWER_TYPES))
    np.random.shuffle(repeated)
    return repeated


def add_group_noise(
    base_ratios: np.ndarray,
    borrower_index: int,
    position_index: int,
    group_count: int,
    max_noise: float = 0.08,
) -> np.ndarray:
    unique_offset = (((borrower_index % 997) / 996) - 0.5) * 0.05
    group_offset = (((position_index + 1) / (group_count + 1)) - 0.5) * 0.20
    month_curve = np.linspace(-0.35, 0.55, num=base_ratios.shape[0])
    structured_noise = group_offset * month_curve
    random_noise = np.random.uniform(-max_noise, max_noise, size=base_ratios.shape[0])
    noise = random_noise + unique_offset + structured_noise
    noise = np.clip(noise, -max_noise, max_noise)
    ratios = base_ratios * (1 + noise)
    return np.minimum.accumulate(ratios)


def generate_stress_profile(
    avg_income: int,
    borrower_index: int,
    position_index: int,
    group_count: int,
    income_ratios: np.ndarray,
    balance_ratios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starting_balance = max(
        avg_income * np.random.uniform(1.1, 2.1),
        np.random.uniform(18000, 95000),
    )
    noisy_income_ratios = add_group_noise(
        income_ratios,
        borrower_index,
        position_index,
        group_count,
    )
    noisy_balance_ratios = add_group_noise(
        balance_ratios,
        borrower_index + 5000,
        position_index,
        group_count,
    )
    incomes = np.round(avg_income * noisy_income_ratios, 2)
    balances = np.round(starting_balance * noisy_balance_ratios, 2)
    return incomes, balances


def generate_healthy_profile(
    avg_income: int,
    borrower_index: int,
    position_index: int,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    starting_balance = max(
        avg_income * np.random.uniform(1.0, 1.8),
        np.random.uniform(15000, 70000),
    )
    base_income = np.array([0.98, 1.00, 1.01, 0.99, 1.02, 1.01], dtype=float)
    base_balance = np.array([0.97, 1.00, 1.02, 0.99, 1.03, 1.01], dtype=float)
    income_noise = add_group_noise(
        base_income,
        borrower_index,
        position_index,
        group_count,
        max_noise=0.02,
    )
    balance_noise = add_group_noise(
        base_balance,
        borrower_index + 7000,
        position_index,
        group_count,
        max_noise=0.03,
    )
    income_ratios = np.clip(income_noise, 0.90, 1.10)
    balance_ratios = np.clip(balance_noise, 0.85, 1.15)
    incomes = np.round(avg_income * income_ratios, 2)
    balances = np.round(starting_balance * balance_ratios, 2)
    return incomes, balances


def build_row(
    borrower_number: int,
    borrower_type: str,
    defaulted: bool,
    incomes: np.ndarray,
    balances: np.ndarray,
    payments: np.ndarray,
) -> dict:
    loan_amount = int(np.random.randint(10000, 200001))
    emi_amount = round(float(loan_amount) * np.random.uniform(0.03, 0.05), 2)
    avg_monthly_income = int(round(float(incomes[0])))

    row = {
        "borrower_id": f"B{borrower_number:04d}",
        "borrower_type": borrower_type,
        "loan_amount": loan_amount,
        "emi_amount": emi_amount,
        "avg_monthly_income": avg_monthly_income,
    }

    for month in range(1, 7):
        row[f"month{month}_balance"] = float(balances[month - 1])
        row[f"month{month}_income"] = float(incomes[month - 1])
        row[f"month{month}_paid"] = bool(payments[month - 1])

    row["defaulted"] = bool(defaulted)
    return row


def build_default_records(starting_borrower_number: int) -> tuple[list[dict], int]:
    records = []
    borrower_number = starting_borrower_number

    for group in STRESS_GROUPS:
        borrower_types = make_even_borrower_types(group["count"])
        for group_index in range(group["count"]):
            avg_income = int(np.random.randint(8000, 80001))
            incomes, balances = generate_stress_profile(
                avg_income=avg_income,
                borrower_index=borrower_number + group_index,
                position_index=group_index,
                group_count=group["count"],
                income_ratios=group["income_ratios"],
                balance_ratios=group["balance_ratios"],
            )
            records.append(
                build_row(
                    borrower_number=borrower_number,
                    borrower_type=str(borrower_types[group_index]),
                    defaulted=True,
                    incomes=incomes,
                    balances=balances,
                    payments=group["payments"],
                )
            )
            borrower_number += 1

    return records, borrower_number


def build_healthy_records(starting_borrower_number: int) -> list[dict]:
    records = []
    borrower_number = starting_borrower_number
    borrower_types = make_even_borrower_types(HEALTHY_COUNT)
    healthy_payments = np.ones(6, dtype=bool)

    for healthy_index in range(HEALTHY_COUNT):
        avg_income = int(np.random.randint(8000, 80001))
        incomes, balances = generate_healthy_profile(
            avg_income,
            borrower_number + healthy_index,
            healthy_index,
            HEALTHY_COUNT,
        )
        records.append(
            build_row(
                borrower_number=borrower_number,
                borrower_type=str(borrower_types[healthy_index]),
                defaulted=False,
                incomes=incomes,
                balances=balances,
                payments=healthy_payments,
            )
        )
        borrower_number += 1

    return records


def main() -> None:
    np.random.seed(42)

    default_records, next_borrower_number = build_default_records(starting_borrower_number=1)
    healthy_records = build_healthy_records(starting_borrower_number=next_borrower_number)

    records = default_records + healthy_records
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Dataset created successfully with 10000 rows")


if __name__ == "__main__":
    main()
