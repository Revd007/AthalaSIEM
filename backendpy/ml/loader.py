"""
Load all .pkl models once at startup and expose predict_threat / predict_anomaly.
Uses joblib; models live in backendpy/models/.
"""
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from app_config import get_model_dir

# Singleton instance
_model_service: Optional["ModelService"] = None


class ModelService:
    """Holds loaded sklearn models and vectorizers; used for inference only."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._rf = None
        self._gb = None
        self._best = None
        self._tfidf = None
        self._label_encoder = None
        self._rag_tfidf = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        # Prefer best_threat_classifier; fallback to random_forest
        best_path = self.model_dir / "best_threat_classifier.pkl"
        rf_path = self.model_dir / "random_forest_model.pkl"
        gb_path = self.model_dir / "gradient_boosting_model.pkl"
        tfidf_path = self.model_dir / "tfidf_vectorizer.pkl"
        le_path = self.model_dir / "label_encoder.pkl"
        rag_path = self.model_dir / "rag_tfidf_vectorizer.pkl"

        if best_path.exists():
            self._best = joblib.load(best_path)
        if rf_path.exists():
            self._rf = joblib.load(rf_path)
        if gb_path.exists():
            self._gb = joblib.load(gb_path)
        if tfidf_path.exists():
            self._tfidf = joblib.load(tfidf_path)
        if le_path.exists():
            self._label_encoder = joblib.load(le_path)
        if rag_path.exists():
            self._rag_tfidf = joblib.load(rag_path)

        self._loaded = True

    @property
    def threat_model(self):
        return self._best if self._best is not None else self._rf

    @property
    def anomaly_model(self):
        return self._gb

    def predict_threat(self, features: np.ndarray) -> tuple[Any, float]:
        """Returns (predicted_class, confidence). Uses best_threat_classifier or random_forest."""
        model = self.threat_model
        if model is None:
            return "unknown", 0.0
        try:
            pred = model.predict(features)
            proba = getattr(model, "predict_proba", None)
            if proba is not None:
                p = proba(features)
                conf = float(np.max(p))
            else:
                conf = 0.0
            label = pred[0] if hasattr(pred, "__getitem__") else pred
            if self._label_encoder is not None:
                label = self._label_encoder.inverse_transform([label])[0]
            return str(label), conf
        except Exception:
            return "unknown", 0.0

    def predict_anomaly(self, features: np.ndarray) -> tuple[float, bool]:
        """Returns (anomaly_score, is_anomaly). Uses gradient_boosting or decision_function if available."""
        model = self.anomaly_model
        if model is None:
            return 0.0, False
        try:
            if hasattr(model, "decision_function"):
                score = float(model.decision_function(features)[0])
            elif hasattr(model, "predict_proba"):
                p = model.predict_proba(features)
                score = float(p[0][1]) if p.shape[1] > 1 else float(p[0][0])
            else:
                pred = model.predict(features)
                score = float(pred[0]) if hasattr(pred, "__getitem__") else float(pred)
            # Simple threshold
            is_anomaly = score > 0.5 or (hasattr(model, "decision_function") and score < -0.1)
            return score, bool(is_anomaly)
        except Exception:
            return 0.0, False

    def get_tfidf_vectorizer(self):
        return self._tfidf

    def get_label_encoder(self):
        return self._label_encoder


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService(get_model_dir())
        _model_service.load()
    return _model_service
