import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go

from app.ml.datasets.loader import load_ieee_train
from app.ml.features.ieee_features import build_features, export_feature_schema

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_PATH = BASE_DIR / "ml_artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)

TARGET_COL = "isFraud"
TRAIN_RATIO = 0.8

# 🔥 РЕЖИМ
FAST_MODE = True

MODEL_CONFIG = {
    "iterations": 1200 if FAST_MODE else 2500,
    "depth": 7,
    "learning_rate": 0.03,
    "l2_leaf_reg": 12,
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 100,

    "task_type": "GPU",
    "devices": "0",
    "bootstrap_type": "Bayesian",
    "grow_policy": "SymmetricTree",
}


# =========================
# LOAD
# =========================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_ieee_train()

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    if FAST_MODE:
        logger.info("⚡ FAST MODE: sampling 150k rows")
        df = df.sample(150_000, random_state=42)

    split = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split], df.iloc[split:]


# =========================
# FEATURES
# =========================
def build_train_val():
    train_raw, val_raw = load_data()

    train_df, cat_cols, selected_v = build_features(train_raw, is_train=True)
    val_df, _, _ = build_features(val_raw, selected_v, is_train=True)

    val_df = val_df[train_df.columns]

    return train_df, val_df, cat_cols


# =========================
# TRAIN
# =========================
def train_model(train_df, val_df, cat_cols):

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL]

    X_val = val_df.drop(columns=[TARGET_COL]).copy()
    y_val = val_df[TARGET_COL]

    # 🔥 FIX типов
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
        if col in X_val.columns:
            X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(**MODEL_CONFIG)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    return model


# =========================
# FEATURE SELECTION
# =========================
def select_features(model, train_df):

    X = train_df.drop(columns=[TARGET_COL])
    importances = model.get_feature_importance()

    features = X.columns

    strong = [
        f for f, imp in zip(features, importances)
        if imp > 0.01
    ]

    logger.info(f"🔥 Selected {len(strong)}/{len(features)} features")

    return strong


# =========================
# RETRAIN
# =========================
def retrain(train_df, val_df, cat_cols, features):

    X_train = train_df[features].copy()
    y_train = train_df[TARGET_COL]

    X_val = val_df[features].copy()
    y_val = val_df[TARGET_COL]

    cat_filtered = [c for c in cat_cols if c in features]

    # 🔥 FIX типов
    for col in cat_filtered:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=cat_filtered)
    val_pool = Pool(X_val, y_val, cat_features=cat_filtered)

    model = CatBoostClassifier(**MODEL_CONFIG)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    return model, cat_filtered


# =========================
# EVALUATE
# =========================
def evaluate(model, val_df, cat_cols):

    X = val_df.drop(columns=[TARGET_COL]).copy()
    y = val_df[TARGET_COL]

    # 🔥 FIX типов
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_cols)

    proba = model.predict_proba(pool)[:, 1]

    pr_auc = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)

    logger.info(f"🔥 PR-AUC: {pr_auc:.4f}")
    logger.info(f"🔥 ROC-AUC: {roc_auc:.4f}")

    # Генерация ROC Curve
    fpr, tpr, thresholds = roc_curve(y, proba)
    fig = go.Figure(data=[go.Scatter(x=fpr, y=tpr, name=f'ROC curve (area = {roc_auc:.4f})', mode='lines', line=dict(color='darkorange', width=2))])
    fig.add_shape(type='line', line=dict(dash='dash', color='navy'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title='ROC Curve - IEEE Transaction Fraud', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=800, height=600)
    
    roc_path = ARTIFACTS_PATH / "roc_curve_ieee.html"
    fig.write_html(str(roc_path))
    logger.info(f"📊 График ROC Curve сохранен в {roc_path}")

    # Генерация PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y, proba)
    fig_pr = go.Figure(data=[go.Scatter(x=recall, y=precision, name=f'PR curve (AP = {pr_auc:.4f})', mode='lines', line=dict(color='purple', width=2))])
    
    # Базовая линия (No Skill) = доля положительных классов
    no_skill = len(y[y == 1]) / len(y)
    fig_pr.add_shape(type='line', line=dict(dash='dash', color='navy'), x0=0, x1=1, y0=no_skill, y1=no_skill)
    fig_pr.update_layout(title='Precision-Recall Curve - IEEE Transaction Fraud', xaxis_title='Recall', yaxis_title='Precision', width=800, height=600)
    
    pr_path = ARTIFACTS_PATH / "pr_curve_ieee.html"
    fig_pr.write_html(str(pr_path))
    logger.info(f"📊 График PR Curve сохранен в {pr_path}")

    return pr_auc


# =========================
# SAVE
# =========================
def save(model, train_df, cat_cols):

    model.save_model(str(ARTIFACTS_PATH / "catboost_final.cbm"))

    export_feature_schema(train_df, cat_cols, ARTIFACTS_PATH)

    logger.info("💾 Model + schema saved")


# =========================
# MAIN
# =========================
def run():

    logger.info("🚀 TRAIN STARTED")

    train_df, val_df, cat_cols = build_train_val()

    # 1. train
    model = train_model(train_df, val_df, cat_cols)

    # 2. feature selection
    features = select_features(model, train_df)

    # 3. retrain
    model, cat_cols = retrain(train_df, val_df, cat_cols, features)

    # 4. evaluate
    evaluate(model, val_df[features + [TARGET_COL]], cat_cols)

    # 5. save
    save(model, train_df[features + [TARGET_COL]], cat_cols)

    logger.info("✅ DONE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()