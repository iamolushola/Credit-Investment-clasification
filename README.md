# Borrower / Investor Classification Pipeline

This project classifies entities into four classes:

1. Prime Borrower
2. Developing Borrower
3. Apex Investor
4. Emerging Investor

The uploaded workbook did not include verified ground-truth class labels. This implementation uses transparent pseudo-label rules based on borrower quality and investor strength. Replace these labels with domain-reviewed labels when available. The packaged artifact uses a Random Forest model selected for speed, explainability, and stable performance; the code also includes Logistic Regression and Gradient Boosting baselines.

## Structure

```text
borrower_investor_ml_project/
├── api.py
├── inference.py
├── train.py
├── src/data_processing.py
├── tests/test_inference.py
├── artifacts/model.joblib
├── artifacts/predictions.csv
├── artifacts/sample_request.json
├── reports/metrics.json
├── reports/classification_report.txt
├── reports/figures/
├── notebooks/borrower_investor_pipeline.ipynb
├── requirements.txt
└── Dockerfile
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Copy your workbook into `data/raw/`, then run:

```bash
python train.py --data data/raw/Machine_Learning_Project\(Sample\).xlsx --sheet "REAL DATA" --fast
```

For a fuller hyperparameter search, remove `--fast`.

If you later add a true label column:

```bash
python train.py --data data/raw/data.xlsx --sheet "REAL DATA" --label-column segment_label
```

## Inference script

```bash
python inference.py --input artifacts/sample_request.json --model artifacts/model.joblib
```

Output contains:

- `entity_id`
- `predicted_class`
- `class_probabilities`
- `confidence_score`
- `explanation`: top feature contributors

## API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"Customer_ID": "CUST-001", "Bureau_Credit_Score": 720}]}'
```

## Tests

```bash
pytest -q
```

## Pseudo-label rules

Two scores are generated:

- **Borrower Quality Score**: credit score, income, low DTI, low utilization, low DPD, low missed payments, low external debt, clean bureau status, and active account status.
- **Investor Strength Score**: investment balance, monthly contribution, accrued returns, investment tenure, portfolio status, and withdrawal ratio.

Rules:

- **Apex Investor**: active investor and high investor strength.
- **Emerging Investor**: active investor and moderate investor strength.
- **Prime Borrower**: high borrower quality where investor profile is weak or absent.
- **Developing Borrower**: remaining borrower profiles.

## Production monitoring

Track:

- Input drift on credit score, DTI, DPD, utilization, investment balance, contribution, and withdrawals.
- Prediction distribution drift across the four classes.
- Confidence distribution drift.
- Per-class precision, recall, and F1 once verified labels are available.
- Manual review rate for predictions with `confidence_score < 0.60`.

Recommended retraining cadence: monthly while label quality improves, then quarterly after the model stabilizes.

## Packaged run metrics

- Holdout macro F1: 0.9165
- Holdout micro F1: 0.9382
- 3-fold CV macro F1: 0.9088 ± 0.0022

See `reports/technical_report.md` and `reports/metrics.json` for details.
