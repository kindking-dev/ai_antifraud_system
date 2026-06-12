import pandas as pd
from pathlib import Path
import structlog
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import json
import plotly.graph_objects as go

logger = structlog.get_logger(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "processed" / "touchalytics_contrastive.parquet"
ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"

def train():
    print("="*50)
    print("🧠 ОБУЧЕНИЕ BEHAVIORAL SIMILARITY MODEL")
    print("="*50)

    if not DATA_PATH.exists():
        logger.error(f"Файл {DATA_PATH} не найден!")
        return

    # 1. Загрузка данных
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Датасет загружен. Размер: {df.shape}")

    X = df.drop(columns=['target'])
    y = df['target']

    # 2. Сплит
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Обучение CatBoost (оптимизировано под быстрый инференс < 5ms)
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        eval_metric='AUC',
        auto_class_weights='Balanced', # Учитываем, что у нас больше 0, чем 1
        random_seed=42,
        od_type='Iter',
        od_wait=50,
        verbose=100
    )

    logger.info("Начало тренировки...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # 4. Метрики для Бизнеса
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print("\n🏆 БИЗНЕС-МЕТРИКИ (ОЦЕНКА КАЧЕСТВА):")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nОтчет классификации:")
    print(classification_report(y_test, y_pred, target_names=["Мошенник (0)", "Владелец (1)"]))

    # 5. Сохранение
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ARTIFACTS_DIR / "behavioral_similarity.cbm"))
    
    # 5.1 Генерация ROC Curve графика (Plotly)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig = go.Figure(data=[go.Scatter(x=fpr, y=tpr, name=f'ROC curve (area = {auc:.4f})', mode='lines', line=dict(color='darkorange', width=2))])
    fig.add_shape(type='line', line=dict(dash='dash', color='navy'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(title='ROC Curve - Behavioral Similarity', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=800, height=600)
    
    roc_path = ARTIFACTS_DIR / "roc_curve_behavioral.html"
    fig.write_html(str(roc_path))
    logger.info(f"📊 График ROC Curve сохранен в {roc_path}")
    
    # 5.2 Генерация PR Curve графика (Plotly)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc_val = average_precision_score(y_test, y_pred_proba)
    fig_pr = go.Figure(data=[go.Scatter(x=recall, y=precision, name=f'PR curve (AP = {pr_auc_val:.4f})', mode='lines', line=dict(color='purple', width=2))])
    # Базовая линия (No Skill) = доля положительных классов
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    fig_pr.add_shape(type='line', line=dict(dash='dash', color='navy'), x0=0, x1=1, y0=no_skill, y1=no_skill)
    fig_pr.update_layout(title='Precision-Recall Curve - Behavioral Similarity', xaxis_title='Recall', yaxis_title='Precision', width=800, height=600)
    
    pr_path = ARTIFACTS_DIR / "pr_curve_behavioral.html"
    fig_pr.write_html(str(pr_path))
    logger.info(f"📊 График PR Curve сохранен в {pr_path}")
    
    schema = {
        "features": list(X.columns),
        "metrics": {"roc_auc": float(auc)},
        "window_size": 15
    }
    with open(ARTIFACTS_DIR / "behavioral_schema.json", "w") as f:
        json.dump(schema, f, indent=4)
        
    logger.info("✅ Модель и схема фичей сохранены в ml_artifacts/")

if __name__ == "__main__":
    train()