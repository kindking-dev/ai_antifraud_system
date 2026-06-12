import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


# =========================
# GLOBAL REGISTRY
# =========================

ml_models: Dict[str, Any] = {}
explainability_models: Dict[str, Any] = {}

MODEL_LOADED: bool = False


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "ml_artifacts" / "catboost_final.cbm"

# (опционально)
SHAP_PATH = BASE_DIR / "ml_artifacts" / "shap_explainer.pkl"


# =========================
# LOAD CORE MODEL
# =========================

def _load_catboost_model() -> CatBoostClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    return model


# =========================
# OPTIONAL: LOAD SHAP
# =========================

def _load_shap_explainer():
    try:
        if SHAP_PATH.exists():
            from app.ml.explainability.shap_explainer import SHAPExplainer

            explainer = SHAPExplainer.load(SHAP_PATH)
            explainability_models["shap_explainer"] = explainer

            logger.info("✅ SHAP explainer loaded")
        else:
            logger.info("ℹ️ SHAP not found (skip)")

    except Exception as e:
        logger.warning(f"⚠️ SHAP load failed: {e}")


# =========================
# PUBLIC LOAD FUNCTION
# =========================

def load_models() -> None:
    """
    Загружает все модели в память.

    Вызывается ОДИН раз при старте FastAPI.
    """

    global MODEL_LOADED

    if MODEL_LOADED:
        logger.info("⚠️ Models already loaded (skip)")
        return

    start = time.perf_counter()

    try:
        logger.info("🚀 Loading ML models...")

        # =========================
        # CORE MODEL
        # =========================
        model = _load_catboost_model()
        ml_models["core_scorer"] = model

        logger.info("✅ CatBoost model loaded")

        # =========================
        # OPTIONAL MODELS
        # =========================
        _load_shap_explainer()

        MODEL_LOADED = True

        logger.info(
            f"🎯 ALL MODELS READY | "
            f"time={round((time.perf_counter() - start)*1000, 2)} ms"
        )

    except Exception as e:
        logger.exception("❌ MODEL LOADING FAILED")
        raise RuntimeError("Model initialization failed") from e


# =========================
# HEALTH CHECK
# =========================

def is_model_ready() -> bool:
    return "core_scorer" in ml_models and MODEL_LOADED


# =========================
# GET MODEL
# =========================

def get_model(name: str = "core_scorer") -> Optional[Any]:
    return ml_models.get(name)


# =========================
# WARMUP
# =========================

def warmup_model() -> None:
    """
    Прогрев модели (убирает cold start latency)
    """

    try:
        if not is_model_ready():
            logger.warning("⚠️ Model not ready for warmup")
            return

        model: CatBoostClassifier = ml_models["core_scorer"]

        import numpy as np
        import pandas as pd

        # dummy input (минимальный)
        dummy = pd.DataFrame([[0.0] * model.feature_count_])

        model.predict_proba(dummy)

        logger.info("🔥 Model warmup completed")

    except Exception as e:
        logger.warning(f"⚠️ Warmup failed: {e}")