# app/ml/inference/transaction_inference.py

import logging
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger(__name__)

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = BASE_DIR / "ml_artifacts" / "catboost_final.cbm"
FEATURE_COLUMNS_PATH = BASE_DIR / "ml_artifacts" / "feature_columns.json"
CAT_FEATURES_PATH = BASE_DIR / "ml_artifacts" / "categorical_features.json"
STATS_PATH = BASE_DIR / "ml_artifacts" / "feature_statistics.json"


# =========================
# LOAD ARTIFACTS
# =========================

def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


FEATURE_COLUMNS: List[str] = _load_json(FEATURE_COLUMNS_PATH)
CAT_FEATURES: List[str] = _load_json(CAT_FEATURES_PATH)
FEATURE_STATS: Dict[str, Any] = _load_json(STATS_PATH)

MEDIANS = FEATURE_STATS.get("medians", {})


# =========================
# LOAD MODEL
# =========================

model = CatBoostClassifier()
MODEL_LOADED = False

try:
    if MODEL_PATH.exists():
        model.load_model(str(MODEL_PATH))
        MODEL_LOADED = True
        logger.info("✅ Transaction model loaded")
    else:
        logger.error(f"❌ Model not found: {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")


# =========================
# FEATURE BUILDER
# =========================
from app.ml.features.ieee_feature_builder import IEEEFeatureBuilder

feature_builder = IEEEFeatureBuilder(artifacts_path=BASE_DIR / "ml_artifacts")

def build_feature_vector(payload: Dict[str, Any], behavior_score: float = 0.5) -> pd.DataFrame:
    return feature_builder.transform_request(payload, behavior_score=behavior_score)


# =========================
# PREDICT
# =========================

def predict_transaction_model(payload: Dict[str, Any], behavior_score: float = 0.5) -> float:
    """
    Returns fraud probability [0..1]
    """

    if not MODEL_LOADED:
        return 0.0

    try:
        X = build_feature_vector(payload, behavior_score=behavior_score)

        pool = Pool(X, cat_features=CAT_FEATURES)

        proba = model.predict_proba(pool)[0][1]

        if np.isnan(proba) or np.isinf(proba):
            return 0.0

        return float(proba)

    except Exception as e:
        logger.error(f"❌ TX prediction error: {e}")
        return 0.0


# =========================
# DEBUG
# =========================

if __name__ == "__main__":
    test_payload = {
        "user_id": "123",
        "amount_kzt": 120.5,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }

    print("TX score:", predict_transaction_model(test_payload))