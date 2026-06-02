# Technical Report: Borrower / Investor Classification Pipeline

## Objective

Build a repeatable classification pipeline that assigns every entity to one of four business segments: Prime Borrower, Developing Borrower, Apex Investor, or Emerging Investor.

## Dataset and label status

The workbook contains 14,638 records from the `REAL DATA` sheet and 44 model-ready features after cleaning and feature engineering. No verified ground-truth target column was present, so the current model was trained on rule-based pseudo-labels. The pseudo-label strategy should be reviewed by credit and investment domain experts before production decisions are automated.

Class distribution:

- Emerging Investor: 4,942
- Apex Investor: 4,473
- Developing Borrower: 3,816
- Prime Borrower: 1,407

## Feature engineering

The pipeline removes direct PII fields such as names, BVN, NIN, phone, email, card account ID, address, and employer name. It then creates model-ready fields including age, address tenure, delinquency severity, historical bureau severity, account-status score, portfolio-status score, credit utilization ratio, external-debt-to-income ratio, contribution-to-income ratio, withdrawal ratio, and returns-to-balance ratio.

## Labeling approach

Two transparent scoring functions were created:

- **Borrower Quality Score**: bureau credit score, monthly income, debt-to-income ratio, credit utilization, days past due, missed payments, external debt, bureau delinquency, and account status.
- **Investor Strength Score**: investment balance, monthly contribution, accrued returns, investment tenure, portfolio status, and withdrawal ratio.

The pseudo-label rules classify active, high-strength investors as Apex Investors; active, moderate-strength investors as Emerging Investors; high-quality non-investor/weak-investor profiles as Prime Borrowers; and the remaining profiles as Developing Borrowers.

## Modeling

Baseline models implemented: Logistic Regression, Random Forest, and Gradient Boosting. The packaged model artifact uses a tuned Random Forest pipeline because it produced strong performance while remaining fast and easy to explain. The training script also includes a slower non-fast mode for cross-validation and hyperparameter search.

## Evaluation summary

Because labels are pseudo-labels, the metrics measure how well the model learns the labeling framework, not real-world credit or investor outcomes.

- Final holdout macro F1: **0.9165**
- Final holdout micro F1: **0.9382**
- Final 3-fold CV macro F1: **0.9088 ± 0.0022**

Per-class holdout F1:

- Apex Investor: 0.9729
- Developing Borrower: 0.9259
- Emerging Investor: 0.9485
- Prime Borrower: 0.8187

## Top predictive features

- num__Investment_Balance (₦): 0.1465
- num__Investment_Withdrawals (₦): 0.1363
- num__Accrued_Returns (₦): 0.1237
- num__Monthly_Contribution (₦): 0.1053
- num__Bureau_Credit_Score: 0.0772
- num__Withdrawal_Ratio: 0.0397
- num__Portfolio_Status_Score: 0.0281
- cat__Investment_Status_-: 0.0261
- num__Contribution_to_Income: 0.0253
- cat__Portfolio_Status_-: 0.0220

SHAP explanations were generated on a sample and saved as `reports/figures/shap_global_importance.png`. Global feature importance is saved as `reports/figures/global_feature_importance.png`.

## Limitations

1. The target labels are pseudo-labels, not verified business outcomes.
2. The investor/borrower split is inferred from available fields, not from an explicit `entity_type` column.
3. Some fields contain placeholder values such as `-`, which were treated as missing.
4. Model performance is likely optimistic because pseudo-labels were generated from the same financial and credit signals used as model features.
5. Production use should require human review for low-confidence predictions and domain validation of labels.

## Next steps

1. Add a verified `segment_label` column from credit/investment experts and retrain with `--label-column`.
2. Confirm business priority: maximize Prime Borrower precision, Prime Borrower recall, or balanced macro F1.
3. Add monthly drift monitoring for credit score, DTI, DPD, utilization, investment balance, contributions, withdrawals, predicted class mix, and confidence distribution.
4. Introduce manual review queues for predictions below 0.60 confidence.
5. Retrain monthly until label quality stabilizes, then move to quarterly retraining.
