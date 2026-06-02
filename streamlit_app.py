"""Streamlit dashboard for local borrower/investor classification.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    clean_and_engineer,
    generate_pseudo_labels,
    make_model_frame,
    split_feature_types,
)

RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path("data/raw/Machine_Learning_Project(Sample).xlsx")
DEFAULT_MODEL_PATH = Path("artifacts/streamlit_model.joblib")

st.set_page_config(
    page_title="Borrower/Investor ML Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
                background-color: #003566;
                color: #ffffff;
            }
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #0077b6 !important;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 8px;
        }
        div[data-testid="metric-container"] .stMetricValue, div[data-testid="metric-container"] .stMetricLabel {
            color: #ffffff !important;
        }
        /* Tables and text */
        .stDataFrame td, .stDataFrame th {
            color: #ffffff !important;
        }
        /* Buttons and links */
        .stButton button, .stDownloadButton button {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                max_categories=30,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, file_name: str, sheet_name: str | None = None):
    suffix = Path(file_name).suffix.lower()
    bio = io.BytesIO(file_bytes)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(bio, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(bio)
    raise ValueError("Upload a .xlsx, .xls, or .csv file")


@st.cache_data(show_spinner=False)
def load_default_data(path_str: str, sheet_name: str = "REAL DATA"):
    path = Path(path_str)
    if not path.exists():
        return None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def find_user_identifier_column(df: pd.DataFrame) -> str | None:
    for candidate in ["Customer_ID", "entity_id", "ID", "User_ID"]:
        if candidate in df.columns:
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def train_model(raw_json: str, model_choice: str, test_size: float):
    raw = pd.read_json(io.StringIO(raw_json), orient="split")
    cleaned = clean_and_engineer(raw)
    labeled = generate_pseudo_labels(cleaned)

    X = make_model_frame(labeled)
    y = labeled["segment_label"].astype(str)
    numeric_features, categorical_features = split_feature_types(X)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    estimators = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimators[model_choice]),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    report_dict = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=pipeline.classes_)

    entity_col = find_user_identifier_column(raw)
    prediction_ids = (
        raw.loc[X_test.index, entity_col].astype(str).values
        if entity_col
        else X_test.index.astype(str)
    )

    bundle = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": pipeline.classes_.tolist(),
        "label_source": "pseudo_labels_rule_based",
        "class_distribution": y.value_counts().to_dict(),
        "user_id_column": entity_col,
    }

    predictions = pd.DataFrame(
        {
            "row_id": X_test.index.astype(str),
            "user_id": prediction_ids,
            "actual_class": y_test.values,
            "predicted_class": pred,
            "confidence_score": proba.max(axis=1),
        }
    )
    for i, cls in enumerate(pipeline.classes_):
        predictions[f"probability_{cls}"] = proba[:, i]

    metrics = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "micro_f1": float(f1_score(y_test, pred, average="micro")),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "labels": pipeline.classes_.tolist(),
        "n_records": int(len(raw)),
        "n_features": int(X.shape[1]),
    }

    labeled_output = labeled.copy()
    all_proba = pipeline.predict_proba(X)
    labeled_output["predicted_class"] = pipeline.predict(X)
    labeled_output["confidence_score"] = all_proba.max(axis=1)
    for i, cls in enumerate(pipeline.classes_):
        labeled_output[f"probability_{cls}"] = all_proba[:, i]

    return bundle, metrics, predictions, labeled_output


def save_model(bundle: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def predict_single(bundle: Dict, raw_df: pd.DataFrame, row_index: int):
    pipeline = bundle["pipeline"]
    row = raw_df.iloc[[row_index]].copy()
    X_row = make_model_frame(row, feature_columns=bundle["feature_columns"])
    proba = pipeline.predict_proba(X_row)[0]
    pred = pipeline.predict(X_row)[0]
    classes = pipeline.classes_.tolist()
    return pred, {classes[i]: float(proba[i]) for i in range(len(classes))}, float(proba.max())


def plot_class_balance(class_distribution: Dict[str, int]):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(class_distribution.keys())
    values = list(class_distribution.values())
    ax.bar(labels, values)
    ax.set_title("Class Balance")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def get_class_table(df: pd.DataFrame, class_name: str, requested_columns: list[str], rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    subset = df.loc[df["predicted_class"] == class_name].copy()
    if subset.empty:
        subset = pd.DataFrame(columns=requested_columns)
    else:
        existing = [col for col in requested_columns if col in subset.columns]
        subset = subset.loc[:, existing].copy()
        for missing in [col for col in requested_columns if col not in subset.columns]:
            subset[missing] = ""
        subset = subset.loc[:, requested_columns]
    if rename_map:
        subset = subset.rename(columns=rename_map)
    return subset


def show_class_summary_cards(df: pd.DataFrame, class_names: list[str]):
    counts = df["predicted_class"].value_counts().reindex(class_names, fill_value=0)
    cols = st.columns(len(class_names))
    # render custom HTML metric cards with specific background color for class summary
    for name, col in zip(class_names, cols):
        count = int(counts.get(name, 0))
        card_html = f"""
        <div style='background-color:#0077b6;padding:12px;border-radius:8px;text-align:center;'>
            <div style='color:#ffffff;font-size:14px;margin-bottom:6px;'>{name}</div>
            <div style='color:#ffffff;font-size:22px;font-weight:700;'>{count}</div>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def plot_numeric_distribution(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    ax.hist(values, bins=35)
    ax.set_title(f"Distribution: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def dataframe_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")


st.title("Borrower/Investor Classification Dashboard")
st.caption("Runs locally on your PC using pandas, scikit-learn, and Streamlit.")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    sheet_name = st.text_input("Excel sheet name", value="REAL DATA")
    model_choice = st.selectbox(
        "Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        index=0,
    )
    test_size = st.slider("Holdout test size", 0.10, 0.40, 0.20, 0.05)
    train_button = st.button("Train / Refresh Model", type="primary")

if uploaded_file is not None:
    raw_df = load_data_from_bytes(uploaded_file.getvalue(), uploaded_file.name, sheet_name or None)
else:
    raw_df = load_default_data(str(DEFAULT_DATA_PATH), sheet_name)

if raw_df is None:
    st.warning(
        "No dataset found. Upload your Excel/CSV file, or place it at "
        "data/raw/Machine_Learning_Project(Sample).xlsx"
    )
    st.stop()

user_id_col = find_user_identifier_column(raw_df)

st.subheader("Dataset Preview")
left, right, third = st.columns(3)
rows_html = f"""
<div style='background-color:#0077b6;padding:12px;border-radius:8px;text-align:center;min-height:72px;display:flex;flex-direction:column;justify-content:center;align-items:center;'>
    <div style='color:#ffffff;font-size:14px;margin-bottom:6px;'>Rows</div>
    <div style='color:#ffffff;font-size:22px;font-weight:700;'>{raw_df.shape[0]:,}</div>
</div>
"""
cols_html = f"""
<div style='background-color:#0077b6;padding:12px;border-radius:8px;text-align:center;min-height:72px;display:flex;flex-direction:column;justify-content:center;align-items:center;'>
    <div style='color:#ffffff;font-size:14px;margin-bottom:6px;'>Columns</div>
    <div style='color:#ffffff;font-size:22px;font-weight:700;'>{raw_df.shape[1]:,}</div>
</div>
"""
source_val = uploaded_file.name if uploaded_file else "Local data/raw file"
source_html = f"""
<div style='background-color:#0077b6;padding:12px;border-radius:8px;text-align:center;min-height:72px;display:flex;flex-direction:column;justify-content:center;align-items:center;'>
    <div style='color:#ffffff;font-size:14px;margin-bottom:6px;'>Source</div>
    <div style='color:#ffffff;font-size:16px;'>{source_val}</div>
</div>
"""
left.markdown(rows_html, unsafe_allow_html=True)
right.markdown(cols_html, unsafe_allow_html=True)
third.markdown(source_html, unsafe_allow_html=True)
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.dataframe(raw_df, use_container_width=True, height=220)

raw_json = raw_df.to_json(orient="split", date_format="iso")

if train_button or "model_bundle" not in st.session_state:
    with st.spinner("Training model and preparing dashboard..."):
        bundle, metrics, predictions, labeled_output = train_model(raw_json, model_choice, test_size)
        st.session_state["model_bundle"] = bundle
        st.session_state["metrics"] = metrics
        st.session_state["predictions"] = predictions
        st.session_state["labeled_output"] = labeled_output
        save_model(bundle, DEFAULT_MODEL_PATH)

bundle = st.session_state["model_bundle"]
metrics = st.session_state["metrics"]
predictions = st.session_state["predictions"]
labeled_output = st.session_state["labeled_output"]

st.divider()
st.subheader("Class Summary")
show_class_summary_cards(labeled_output, ["Prime Borrower", "Developing Borrower", "Apex Investor", "Emerging Investor"])

st.subheader("Class Member Tables")
class_table_specs = {
    "Apex Investor": {
        "columns": [
            "First_Name",
            "Last_Name",
            "Investment_Balance (₦)",
            "Product_Type",
            "Investment_Tenure (Days)",
            "Portfolio_Status",
        ],
        "rename": {"Investment_Balance (₦)": "AUM(₦)"},
    },
    "Emerging Investor": {
        "columns": [
            "First_Name",
            "Last_Name",
            "Investment_Balance (₦)",
            "Product_Type",
            "Investment_Tenure (Days)",
            "Portfolio_Status",
        ],
        "rename": {"Investment_Balance (₦)": "AUM(₦)"},
    },
    "Prime Borrower": {
        "columns": [
            "First_Name",
            "Last_Name",
            "Employment_Status",
            "Employer_Name",
            "Industry_Sector",
            "Credit_Limit_Assigned",
            "Investment_Status",
        ],
        "rename": {},
    },
    "Developing Borrower": {
        "columns": [
            "First_Name",
            "Last_Name",
            "Employment_Status",
            "Employer_Name",
            "Industry_Sector",
            "Credit_Limit_Assigned",
            "Investment_Status",
        ],
        "rename": {},
    },
}
for class_name, spec in class_table_specs.items():
    st.markdown(f"### {class_name}")
    table = get_class_table(labeled_output, class_name, spec["columns"], spec["rename"])
    if table.empty:
        st.write("No records found for this class.")
    else:
        st.dataframe(table.reset_index(drop=True), use_container_width=True, height=220)

st.subheader("Predictions")
st.dataframe(predictions, use_container_width=True, height=220)
col_a, col_b = st.columns(2)
with col_a:
    dataframe_download(predictions, "holdout_predictions.csv", "Download Holdout Predictions")
with col_b:
    dataframe_download(labeled_output, "all_records_predictions.csv", "Download All Predictions")

st.subheader("Per-Class Report")
report_df = pd.DataFrame(metrics["classification_report"]).T
st.dataframe(report_df, use_container_width=True, height=220)

st.subheader("Saved Artifacts")
st.write(f"Model saved to: `{DEFAULT_MODEL_PATH}`")
st.write("To use this model later, load it with `joblib.load('artifacts/streamlit_model.joblib')`.")
