# Stelvora

Post-disbursement loan health infrastructure for Indian NBFCs.

Stelvora monitors every active loan daily, predicts borrower stress before any EMI is missed, and automatically triggers intervention to prevent defaults.

## What This Prototype Does

- Generates synthetic Indian NBFC loan data for 10,000 borrowers
- Calculates a Loan Health Score using 5 financial signals
- Identifies stressed borrowers across Green, Yellow, Orange, and Red risk buckets
- Shows borrower-level prediction timelines with early warning detection
- Runs a shadow pilot analysis to estimate NPA prevention impact

## How To Run

```bash
pip install pandas numpy scikit-learn streamlit plotly
python generate_data.py
python model.py
streamlit run dashboard.py
