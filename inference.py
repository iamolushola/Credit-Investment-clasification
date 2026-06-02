"""Local inference script for borrower/investor classification."""
from __future__ import annotations
import argparse, json, logging, warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Any, Dict, List
import joblib, numpy as np, pandas as pd
from src.data_processing import make_model_frame
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

def load_model(model_path: str | Path = "artifacts/model.joblib") -> Dict[str, Any]:
    return joblib.load(model_path)

def _read_records(input_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict): return [payload]
        if isinstance(payload, list): return payload
        raise ValueError("JSON input must be an object or list of objects")
    if path.suffix.lower() == ".csv": return pd.read_csv(path).to_dict(orient="records")
    if path.suffix.lower() in {".xlsx", ".xls"}: return pd.read_excel(path).to_dict(orient="records")
    raise ValueError("Supported input formats: .json, .csv, .xlsx")

def explain_prediction(bundle: Dict[str, Any], X_prepared: pd.DataFrame, row_index: int = 0, top_n: int = 5):
    pipeline = bundle["pipeline"]; pre = pipeline.named_steps["preprocess"]; est = pipeline.named_steps["model"]
    names = pre.get_feature_names_out().tolist() if hasattr(pre, "get_feature_names_out") else bundle.get("feature_names", [])
    Xt = pre.transform(X_prepared.iloc[[row_index]])
    row_values = Xt.toarray()[0] if hasattr(Xt, "toarray") else np.asarray(Xt)[0]
    importances = getattr(est, "feature_importances_", np.ones(len(row_values)))
    impact = np.abs(row_values) * np.asarray(importances)
    top_idx = np.argsort(impact)[::-1][:top_n]
    return [{"feature": names[i] if i < len(names) else f"feature_{i}", "direction": "+" if row_values[i] >= 0 else "-", "magnitude": float(impact[i])} for i in top_idx]

def predict_records(records: List[Dict[str, Any]], model_path: str | Path = "artifacts/model.joblib") -> List[Dict[str, Any]]:
    bundle = load_model(model_path); pipeline = bundle["pipeline"]; raw = pd.DataFrame(records)
    X = make_model_frame(raw, feature_columns=bundle["feature_columns"])
    proba = pipeline.predict_proba(X); preds = pipeline.predict(X); classes = pipeline.classes_.tolist()
    entity_ids = raw["Customer_ID"].astype(str).tolist() if "Customer_ID" in raw.columns else [str(i) for i in range(len(raw))]
    results = []
    for i, pred in enumerate(preds):
        results.append({"entity_id": entity_ids[i], "predicted_class": str(pred), "class_probabilities": {cls: float(proba[i, j]) for j, cls in enumerate(classes)}, "confidence_score": float(proba[i].max()), "explanation": explain_prediction(bundle, X, row_index=i, top_n=5)})
    return results

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args(); records = _read_records(args.input); print(json.dumps(predict_records(records, args.model), indent=2))
if __name__ == "__main__": main()
