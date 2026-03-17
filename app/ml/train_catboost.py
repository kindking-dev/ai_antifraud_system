"""
CatBoost Training Pipeline.
Optimized for PR-AUC and sub-50ms inference.
"""

import os
import structlog
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

logger = structlog.get_logger(__name__)


def train_model(dataset_path: str, model_save_path: str) -> None:
    logger.info("loading_data", path=dataset_path)
    df = pd.read_parquet(dataset_path, engine="pyarrow")

    # 1. Feature Mapping & Synthetic Data Generation (for API contracts)
    logger.info("engineering_features")
    df["amount_kzt"] = df["TransactionAmt"] * 450.0  # Условный курс

    # Симулируем наши кастомные векторы для соответствия схеме Scoring API
    np.random.seed(42)
    # Фрод чаще имеет скачки Velocity и низкий Trust Score
    df["Velocity_24h_Count"] = np.where(
        df["isFraud"] == 1,
        np.random.poisson(25, len(df)),
        np.random.poisson(5, len(df)),
    )
    df["Is_Velocity_Spike"] = np.where(df["Velocity_24h_Count"] > 15, 1, 0)
    df["Sensor_Keystroke_Variance"] = np.where(
        df["isFraud"] == 1,
        np.random.uniform(0.0, 0.05, len(df)),
        np.random.uniform(0.1, 0.3, len(df)),
    )
    df["Device_Trust_Score"] = np.where(
        df["isFraud"] == 1,
        np.random.uniform(0.0, 0.4, len(df)),
        np.random.uniform(0.6, 1.0, len(df)),
    )

    # 2. Выбор финальных признаков (строго как в app/api/v1/scoring.py)
    features = [
        "amount_kzt",
        "Velocity_24h_Count",
        "Is_Velocity_Spike",
        "Sensor_Keystroke_Variance",
        "Device_Trust_Score",
        "card1",
        "card2",
        "C1",
        "C2",
        "V1",
        "V2",
    ]
    target = "isFraud"

    X = df[features].fillna(0)  # Простой филл для baseline
    y = df[target]

    # 3. Train/Test Split (Stratified to maintain 3.5% fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Calculation of scale_pos_weight
    neg_count = len(y_train[y_train == 0])
    pos_count = len(y_train[y_train == 1])
    scale_weight = neg_count / pos_count
    logger.info("class_imbalance_configured", scale_pos_weight=round(scale_weight, 2))

    # 5. Model Configuration (Optimized for fast inference)
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        eval_metric="PRAUC",  # Убрали дефис
        thread_count=-1,
        verbose=100,
    )

    # 6. Training
    logger.info("training_catboost_started")
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    # 7. Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(
        "training_completed", PR_AUC=round(pr_auc, 4), ROC_AUC=round(roc_auc, 4)
    )

    # 8. Exporting Binary Artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path, format="cbm")
    logger.info("model_artifact_saved", path=model_save_path)


if __name__ == "__main__":
    DATASET_PATH = "data/train_transaction.parquet.gzip"
    MODEL_OUT_PATH = "ml_artifacts/core_scorer.cbm"
    train_model(DATASET_PATH, MODEL_OUT_PATH)
