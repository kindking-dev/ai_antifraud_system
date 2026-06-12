"""
SENTINEL AI: Unified IEEE-CIS Feature Builder.
Ensures 100% parity between Training and Production Inference.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class IEEEFeatureBuilder:
    def __init__(self, artifacts_path: Optional[Path] = None):
        """
        Инициализация билдера. Загружает схемы и статистики для инференса.
        """
        self.feature_columns = []
        self.cat_cols = []
        self.stats = {}
        self.freq_maps = {}
        self.is_ready = False

        if artifacts_path:
            self._load_artifacts(artifacts_path)

    def _load_artifacts(self, path: Path):
        """Загружает JSON-артефакты обучения."""
        try:
            with open(path / "feature_columns.json", "r") as f:
                self.feature_columns = json.load(f)
            with open(path / "categorical_features.json", "r") as f:
                self.cat_cols = json.load(f)
            with open(path / "feature_statistics.json", "r") as f:
                self.stats = json.load(f)

            freq_path = path / "frequency_maps.json"
            if freq_path.exists():
                with open(freq_path, "r") as f:
                    self.freq_maps = json.load(f)

            self.is_ready = True
            logger.info(
                f"✅ Feature Builder ready with {len(self.feature_columns)} features"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")

    def transform_request(
        self, payload: Dict[str, Any], behavior_score: float = 0.5
    ) -> pd.DataFrame:
        """
        Превращает один API-запрос в вектор для CatBoost.
        """
        # Превращаем JSON в DataFrame (1 строка)
        df = pd.DataFrame([payload])

        # Map fields
        if "amount_kzt" in df.columns and "TransactionAmt" not in df.columns:
            df["TransactionAmt"] = df["amount_kzt"]
        if "TransactionAmt" not in df.columns:
            df["TransactionAmt"] = 0.0
            
        if "timestamp_utc" in df.columns and "TransactionDT" not in df.columns:
            from datetime import datetime, timezone
            try:
                ts = datetime.fromisoformat(str(df["timestamp_utc"].iloc[0]).replace("Z", "+00:00"))
                df["TransactionDT"] = ts.timestamp()
            except Exception:
                df["TransactionDT"] = datetime.now(timezone.utc).timestamp()
        if "TransactionDT" not in df.columns:
            from datetime import datetime, timezone
            df["TransactionDT"] = datetime.now(timezone.utc).timestamp()

        # 1. Temporal Engineering (TransactionDT -> hour, dow, etc.)
        df = self._add_time_features(df)

        # 2. Amount Engineering (log, z-score, mean_ratio)
        df = self._add_amount_features(df)

        # 3. Frequency Encoding (используем карты из обучения)
        for col in ["card1", "addr1"]:
            val = str(payload.get(col, "unknown"))
            df[f"{col}_freq"] = self.freq_maps.get(col, {}).get(val, 0.0)

        # 4. 🔥 LATE FUSION: Инъекция скора биометрии
        if "behavior_score" in self.feature_columns:
            df["behavior_score"] = float(behavior_score)

        # 5. Align & Fill (строгое соблюдение контракта признаков)
        # Создаем пустой DF с нужными колонками
        final_df = pd.DataFrame(index=[0], columns=self.feature_columns)

        for col in self.feature_columns:
            if col in df.columns:
                final_df[col] = df[col].iloc[0]
            else:
                # Наполнение отсутствующих V-фич медианами
                final_df[col] = self.stats.get("medians", {}).get(col, 0.0)

        # 6. Типизация для CatBoost
        for col in self.feature_columns:
            if col in self.cat_cols:
                final_df[col] = str(final_df[col].iloc[0])
            else:
                final_df[col] = np.float32(final_df[col].iloc[0])

        return final_df[self.feature_columns]

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет временных циклов."""
        dt = float(df["TransactionDT"].iloc[0])
        hour = (dt / 3600) % 24
        dow = (dt // 86400) % 7

        df["hour"] = np.int8(hour)
        df["day_of_week"] = np.int8(dow)
        df["is_night"] = np.int8(1 if hour <= 6 else 0)
        df["is_weekend"] = np.int8(1 if dow >= 5 else 0)
        df["days_since_start"] = np.float32(dt / 86400)
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Статистический инжиниринг суммы транзакции."""
        amt = float(df["TransactionAmt"].iloc[0])
        df["log_amount"] = np.float32(np.log1p(amt))

        # Используем константы обучения (μ и σ)
        mean_global = self.stats.get("means", {}).get("TransactionAmt", amt)
        std_global = self.stats.get("stds", {}).get("TransactionAmt", 1.0)

        df["amount_to_mean"] = np.float32(amt / (mean_global + 1e-3))
        df["amount_zscore"] = np.float32((amt - mean_global) / (std_global + 1e-3))
        df["amount_log_ratio"] = np.float32(
            df["log_amount"] / (np.log1p(mean_global) + 1e-3)
        )
        return df

    @staticmethod
    def generate_frequency_maps(
        train_df: pd.DataFrame, cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Генерация карт частот (вызывается только при обучении)."""
        maps = {}
        for col in cols:
            maps[col] = train_df[col].value_counts(normalize=True).to_dict()
        return maps
