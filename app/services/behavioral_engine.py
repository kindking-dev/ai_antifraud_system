import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)

BEHAVIORAL_COLS = [
    "duration_ms_mean",
    "duration_ms_std",
    "duration_ms_max",
    "length_px_mean",
    "length_px_std",
    "length_px_max",
    "velocity_mean",
    "velocity_std",
    "velocity_max",
    "median_pressure_mean",
    "median_pressure_std",
    "median_pressure_max",
    "median_area_mean",
    "median_area_std",
    "median_area_max",
]

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "ml_artifacts" / "behavioral_similarity.cbm"

class BehavioralEngine:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.model_loaded = False
        try:
            if MODEL_PATH.exists():
                self.model.load_model(str(MODEL_PATH))
                self.model_loaded = True
            else:
                logger.warning(f"⚠️ Model not found at {MODEL_PATH}, will use fallback heuristic.")
        except Exception as e:
            logger.error(f"❌ Failed to load behavioral model: {e}")

    def get_defaults_dict(self) -> Dict[str, float]:
        """Provides a default baseline profile when none exists in Redis."""
        return {col: 0.0 for col in BEHAVIORAL_COLS}

    def extract_profile_from_events(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extacts behavioral features from raw touch events.
        """
        if not events or len(events) < 2:
            return {col: 0.0 for col in BEHAVIORAL_COLS}
            
        durations = []
        lengths = []
        velocities = []
        pressures = []
        areas = []
        
        for i in range(1, len(events)):
            e1 = events[i-1]
            e2 = events[i]
            
            dt = max(1, e2.get("timestamp_ms", 0) - e1.get("timestamp_ms", 0))
            dx = e2.get("x_pos", 0) - e1.get("x_pos", 0)
            dy = e2.get("y_pos", 0) - e1.get("y_pos", 0)
            
            length = np.sqrt(dx**2 + dy**2)
            velocity = length / dt
            
            durations.append(dt)
            lengths.append(length)
            velocities.append(velocity)
            pressures.append(e2.get("pressure", 0.0))
            areas.append(e2.get("contact_size", 0.0))
            
        if not durations:
            return {col: 0.0 for col in BEHAVIORAL_COLS}
            
        def safe_std(arr):
            return float(np.std(arr)) if len(arr) > 1 else 0.0
            
        profile = {
            "duration_ms_mean": float(np.mean(durations)),
            "duration_ms_std": safe_std(durations),
            "duration_ms_max": float(np.max(durations)),
            
            "length_px_mean": float(np.mean(lengths)),
            "length_px_std": safe_std(lengths),
            "length_px_max": float(np.max(lengths)),
            
            "velocity_mean": float(np.mean(velocities)),
            "velocity_std": safe_std(velocities),
            "velocity_max": float(np.max(velocities)),
            
            "median_pressure_mean": float(np.mean(pressures)),
            "median_pressure_std": safe_std(pressures),
            "median_pressure_max": float(np.max(pressures)),
            
            "median_area_mean": float(np.mean(areas)),
            "median_area_std": safe_std(areas),
            "median_area_max": float(np.max(areas)),
        }
        
        return profile
        
    def calculate_risk(self, baseline_profile: Dict[str, float], current_profile: Dict[str, float]) -> float:
        """
        Calculates risk (0.0 to 1.0).
        1.0 means High Risk (Hacker), 0.0 means Low Risk (True Owner).
        """
        input_vector = []
        for col in BEHAVIORAL_COLS:
            base = baseline_profile.get(col, 0.0)
            curr = current_profile.get(col, 0.0)
            input_vector.append(abs(curr - base))
            
        if self.model_loaded:
            # Model predicts probability. In behavioral_pipeline, target 0 = Fraud
            try:
                pred_proba = self.model.predict_proba([input_vector])
                fraud_prob = float(pred_proba[0][0])
                return fraud_prob
            except Exception as e:
                logger.error(f"Inference error: {e}")
                
        # Fallback heuristic if model is not loaded or failed
        distances = []
        for i, col in enumerate(BEHAVIORAL_COLS):
            base = max(1.0, baseline_profile.get(col, 1.0))
            diff = input_vector[i] / base
            distances.append(diff)
        
        mean_diff = float(np.mean(distances))
        risk = 1.0 / (1.0 + np.exp(-10 * (mean_diff - 0.35)))
        return float(risk)