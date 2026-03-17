"""
CatBoost Training Pipeline with SMOTE-Tomek balancing.
Optimized for high Precision-Recall AUC and strict inference constraints.
"""

import sys
from pathlib import Path

import pandas as pd
import structlog
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(sys.stdout),
)
logger = structlog.get_logger(__name__)


def train_model() -> None:
    """Executes the training pipeline, applies sampling, and saves the artifact."""
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "train_processed.parquet"
    artifact_dir = project_root / "ml_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    
    if not data_path.exists():
        logger.error("processed_data_missing", path=str(data_path))
        sys.exit(1)

    logger.info("loading_training_data", path=str(data_path))
    df = pd.read_parquet(data_path, engine='pyarrow')

    # Define core feature subset for the MVP (combining financial + our synthetic ones)
    core_features = [
        'TransactionAmt_Local', 'Velocity_24h_Count', 'Is_Velocity_Spike',
        'Sensor_Keystroke_Variance', 'Device_Trust_Score', 
        'card1', 'card2', 'C1', 'C2', 'V1', 'V2' # Sample of native IEEE-CIS numeric features
    ]
    target = 'isFraud'

    # Drop rows where target is missing, just in case
    df = df.dropna(subset=[target])
    
    X = df[core_features].copy()
    y = df[target].copy()

    # Imputation: SMOTE-Tomek requires dense data without NaNs.
    # We use median imputation for robust handling of outliers.
    X.fillna(X.median(), inplace=True)

    logger.info("data_prepared", x_shape=X.shape, class_distribution=y.value_counts().to_dict())

    # Split before SMOTE to prevent data leakage into the validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE-Tomek ONLY on training data
    logger.info("applying_smote_tomek")
    smt = SMOTETomek(random_state=42, n_jobs=-1)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
    logger.info("resampling_complete", new_distribution=pd.Series(y_train_resampled).value_counts().to_dict())

    # Initialize CatBoost. 
    # Rationale: depth=6 and iterations=500 keep the model small and fast for the <50ms SLA.
    clf = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        eval_metric='AUC', # Fallback eval metric, we calculate PR-AUC manually
        thread_count=-1,
        random_seed=42,
        verbose=50
    )

    logger.info("training_catboost")
    clf.fit(X_train_resampled, y_train_resampled, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Evaluate using PR-AUC
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_pred_proba)
    
    logger.info("evaluation_completed", pr_auc=round(pr_auc, 4))

    # Save artifact
    model_path = artifact_dir / "catboost_core_v1.cbm"
    clf.save_model(str(model_path))
    logger.info("model_saved", path=str(model_path))


if __name__ == "__main__":
    train_model()