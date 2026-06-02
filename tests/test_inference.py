import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import predict_records


def test_predict_single_record():
    sample_path = ROOT / "artifacts" / "sample_request.json"
    model_path = ROOT / "artifacts" / "model.joblib"
    assert sample_path.exists(), "Run train.py first to create artifacts/sample_request.json"
    record = json.loads(sample_path.read_text())
    result = predict_records([record], model_path=model_path)[0]
    assert "predicted_class" in result
    assert "class_probabilities" in result
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-6
    assert result["confidence_score"] >= 0
    assert isinstance(result["explanation"], list)
