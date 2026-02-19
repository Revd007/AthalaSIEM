"""
Build feature vector from a log row for ML inference.
Uses TF-IDF from loader plus numeric/categorical from log fields.
"""
from typing import Any, Optional

import numpy as np

from ml.loader import get_model_service


def _level_to_int(level: Optional[str]) -> int:
    if not level:
        return 0
    m = {"critical": 4, "error": 3, "warning": 2, "information": 1, "info": 1, "debug": 0}
    return m.get(level.lower().strip(), 0)


def _source_hash(source: Optional[str]) -> int:
    if not source:
        return 0
    return hash(source) % 10000


def extract_features(
    message: str,
    level: str = "",
    source: str = "",
    event_id: int = 0,
    category: Optional[str] = None,
) -> np.ndarray:
    """
    Produce a feature vector for the threat/anomaly models.
    Uses TF-IDF of message when vectorizer exists (matches typical training);
    otherwise a small numeric vector.
    """
    svc = get_model_service()
    tfidf = svc.get_tfidf_vectorizer()

    if tfidf is not None:
        try:
            X = tfidf.transform([message or ""])
            if hasattr(X, "toarray"):
                X = X.toarray()
            return X.astype(np.float64)
        except Exception:
            pass

    # Fallback: numeric features only
    level_val = _level_to_int(level)
    source_val = _source_hash(source)
    msg_len = min(len(message or ""), 10000)
    event_id_val = int(event_id) if event_id else 0
    cat_hash = hash(category or "") % 10000
    return np.array(
        [[level_val, source_val, msg_len, event_id_val, cat_hash]],
        dtype=np.float64,
    )
