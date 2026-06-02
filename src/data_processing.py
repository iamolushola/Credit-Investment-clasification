"""Data cleaning, feature engineering, and pseudo-label generation."""
from __future__ import annotations
import re
from typing import Iterable, Optional
import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2026-06-01")
PII_COLUMNS = ["Customer_ID","First_Name","Last_Name","BVN","NIN","Primary_Phone","Email_Address","Residential_Address","Card_Account_ID","Primary_Bank_NUBAN","Employer_Name"]
LABEL_COLUMNS = ["segment_label", "Investor_Strength_Score", "Borrower_Quality_Score"]
DATE_RAW_COLUMNS = ["Date_of_Birth"]
TEXT_DURATION_COLUMNS = ["Duration at Current Address"]
INVESTMENT_NUMERIC_COLUMNS = ["Investment_Balance (₦)","Monthly_Contribution (₦)","Investment_Tenure (Days)","Expected_Return_Rate (%)","Accrued_Returns (₦)","Investment_Withdrawals (₦)"]
GENERAL_NUMERIC_COLUMNS = ["Number of Dependants","Monthly_Net_Income","Income_Verify_Method","Bureau_Credit_Score","Active_External_Loans","External_Debt_Owed","Debt_to_Income_Ratio","Credit_Limit_Assigned","Current_Utilized_Amt","Days_Past_Due (DPD)","Missed_Payments_Count"]

def parse_numeric(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float, np.number)): return float(value)
    text = str(value).strip()
    if text in {"-", "", "nan", "None", "N/A"}: return np.nan
    text = text.replace("₦", "").replace(",", "").replace("%", "")
    try: return float(text)
    except ValueError: return np.nan

def extract_years(value):
    if pd.isna(value): return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan

def delinquency_severity(value):
    if pd.isna(value) or str(value).strip() in {"", "-", "nan"}: return 0
    text = str(value).lower()
    if "write" in text or "off" in text: return 5
    if "180" in text: return 4
    if "90" in text: return 3
    if "60" in text: return 2
    if "30" in text: return 1
    if "current" in text: return 0
    return 0

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop(columns=[c for c in data.columns if str(c).startswith("Unnamed")], errors="ignore")
    for col in INVESTMENT_NUMERIC_COLUMNS:
        if col in data.columns: data[col] = data[col].map(parse_numeric)
    for col in GENERAL_NUMERIC_COLUMNS:
        if col in data.columns: data[col] = pd.to_numeric(data[col], errors="coerce")
    if "Date_of_Birth" in data.columns:
        dob = pd.to_datetime(data["Date_of_Birth"], errors="coerce")
        data["Age"] = ((REFERENCE_DATE - dob).dt.days / 365.25).clip(lower=18, upper=80)
    if "Duration at Current Address" in data.columns:
        data["Address_Tenure_Years"] = data["Duration at Current Address"].map(extract_years)
    for col in ["Bureau_Delinquency", "Worst_Historical_Status"]:
        if col in data.columns: data[f"{col}_Severity"] = data[col].map(delinquency_severity)
    if "Account_Status" in data.columns:
        account_map = {"active": 2, "watch": 1, "delinquent": 0, "closed": 0}
        data["Account_Status_Score"] = data["Account_Status"].astype(str).str.lower().map(account_map).fillna(1)
    if "Portfolio_Status" in data.columns:
        portfolio_map = {"growing": 3, "stable": 2, "declining": 1, "-": 0}
        data["Portfolio_Status_Score"] = data["Portfolio_Status"].astype(str).str.lower().map(portfolio_map).fillna(0)
    def safe_ratio(num, den):
        return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    if {"Current_Utilized_Amt", "Credit_Limit_Assigned"}.issubset(data.columns):
        data["Credit_Utilization_Ratio"] = safe_ratio(data["Current_Utilized_Amt"], data["Credit_Limit_Assigned"])
    if {"External_Debt_Owed", "Monthly_Net_Income"}.issubset(data.columns):
        data["External_Debt_to_Income"] = safe_ratio(data["External_Debt_Owed"], data["Monthly_Net_Income"])
    if {"Monthly_Contribution (₦)", "Monthly_Net_Income"}.issubset(data.columns):
        data["Contribution_to_Income"] = safe_ratio(data["Monthly_Contribution (₦)"], data["Monthly_Net_Income"])
    if {"Investment_Withdrawals (₦)", "Investment_Balance (₦)"}.issubset(data.columns):
        data["Withdrawal_Ratio"] = safe_ratio(data["Investment_Withdrawals (₦)"], data["Investment_Balance (₦)"])
    if {"Accrued_Returns (₦)", "Investment_Balance (₦)"}.issubset(data.columns):
        data["Returns_to_Balance"] = safe_ratio(data["Accrued_Returns (₦)"], data["Investment_Balance (₦)"])
    for col in ["Credit_Utilization_Ratio","External_Debt_to_Income","Contribution_to_Income","Withdrawal_Ratio","Returns_to_Balance"]:
        if col in data.columns: data[col] = data[col].replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    return data

def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    fill = numeric.median() if numeric.notna().any() else 0
    return numeric.fillna(fill).rank(pct=True, ascending=ascending).fillna(0.5)

def generate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    inv_status = data.get("Investment_Status", pd.Series("", index=data.index)).astype(str).str.strip().str.lower()
    aum = data.get("Investment_Balance (₦)", pd.Series(0, index=data.index)).fillna(0)
    monthly_contribution = data.get("Monthly_Contribution (₦)", pd.Series(0, index=data.index)).fillna(0)
    tenure_days = data.get("Investment_Tenure (Days)", pd.Series(0, index=data.index)).fillna(0)
    num_products = pd.to_numeric(data.get("Number_of_Investment_Products", pd.Series(0, index=data.index)), errors="coerce").fillna(0)
    portfolio_status = data.get("Portfolio_Status", pd.Series("", index=data.index)).astype(str).str.strip().str.lower()

    is_active = inv_status.eq("active")
    is_inactive_or_closed = inv_status.isin(["inactive", "closed"])
    portfolio_good = portfolio_status.isin(["growing", "stable"])
    portfolio_declining = portfolio_status.eq("declining")

    apex_rule_1 = (aum >= 50_000_000) & is_active & portfolio_good
    apex_rule_2 = (
        (aum >= 20_000_000)
        & (monthly_contribution >= 500_000)
        & (tenure_days >= 365)
        & (num_products >= 2)
        & is_active
    )
    is_apex_investor = apex_rule_1 | apex_rule_2
    is_emerging_investor = is_inactive_or_closed | portfolio_declining | (aum < 20_000_000)

    # Borrower classification with hard overrides
    dti = pd.to_numeric(data.get("Debt_to_Income_Ratio", pd.Series(np.nan, index=data.index)), errors="coerce").fillna(50)
    worst_status = data.get("Worst_Historical_Status", pd.Series("", index=data.index)).astype(str).str.strip().str.lower()
    missed_payments = pd.to_numeric(data.get("Missed_Payments_Count", pd.Series(0, index=data.index)), errors="coerce").fillna(0)
    credit_score = pd.to_numeric(data.get("Bureau_Credit_Score", pd.Series(0, index=data.index)), errors="coerce").fillna(0)

    # Hard override conditions for Developing Borrower
    hard_override_dti = dti > 45
    hard_override_status = worst_status.isin(["default", "charged off"])
    hard_override_payments = missed_payments > 5
    hard_override_score = credit_score < 550
    has_hard_override = hard_override_dti | hard_override_status | hard_override_payments | hard_override_score

    # Calculate borrower quality score (0-100)
    borrower_quality = (
        0.28 * _rank_pct(data.get("Bureau_Credit_Score", pd.Series(np.nan, index=data.index))) * 100
        + 0.12 * _rank_pct(data.get("Monthly_Net_Income", pd.Series(np.nan, index=data.index))) * 100
        + 0.12 * (1 - _rank_pct(data.get("Debt_to_Income_Ratio", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.10 * (1 - _rank_pct(data.get("Credit_Utilization_Ratio", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.12 * (1 - _rank_pct(data.get("Days_Past_Due (DPD)", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.08 * (1 - _rank_pct(data.get("Missed_Payments_Count", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.08 * (1 - _rank_pct(data.get("External_Debt_to_Income", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.05 * (1 - _rank_pct(data.get("Worst_Historical_Status_Severity", pd.Series(np.nan, index=data.index)), ascending=True)) * 100
        + 0.05 * _rank_pct(data.get("Account_Status_Score", pd.Series(np.nan, index=data.index))) * 100
    )

    # Prime Borrower: score >= 70 AND no hard overrides
    is_prime_borrower = (borrower_quality >= 70) & ~has_hard_override
    is_developing_borrower = ~is_prime_borrower

    labels = np.where(
        is_apex_investor,
        "Apex Investor",
        np.where(
            is_emerging_investor,
            "Emerging Investor",
            np.where(is_prime_borrower, "Prime Borrower", "Developing Borrower"),
        ),
    )

    data["Investor_Strength_Score"] = aum
    data["Borrower_Quality_Score"] = borrower_quality
    data["segment_label"] = labels
    return data

def make_model_frame(df: pd.DataFrame, feature_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    data = clean_and_engineer(df)
    drop_columns = set(PII_COLUMNS + LABEL_COLUMNS + DATE_RAW_COLUMNS + TEXT_DURATION_COLUMNS)
    X = data.drop(columns=[c for c in drop_columns if c in data.columns], errors="ignore")
    if feature_columns is not None:
        for col in feature_columns:
            if col not in X.columns: X[col] = np.nan
        X = X[list(feature_columns)]
    return X

def split_feature_types(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features
