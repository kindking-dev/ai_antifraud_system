import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import structlog

# Настройка логирования
logging_config = structlog.get_logger(__name__)
logger = logging_config.bind(stage="EDA")

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
# Укажи здесь точный путь, куда ты положил скачанный CSV
RAW_DATA_PATH = Path(r"C:\Users\Islam\Desktop\Diploma\ai_antifraud_system\data\raw\raw_touches.csv")
REPORT_DIR = BASE_DIR / "reports" / "touchalytics_eda"

# Имена колонок по спецификации Mario Frank
COLUMNS = [
    'phone_id', 'user_id', 'doc_id', 'time_ms', 'action', 
    'phone_orient', 'x', 'y', 'pressure', 'area', 'finger_orient'
]

def load_data() -> pd.DataFrame:
    logger.info("Загрузка сырых данных (50MB)...")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH, names=COLUMNS, header=0, on_bad_lines='skip')
    logger.info(f"Загружено {len(df):,} строк.")
    return df

def basic_statistics(df: pd.DataFrame):
    logger.info("Расчет базовых статистик...")
    
    stats = {
        "Total Rows": len(df),
        "Unique Users": df['user_id'].nunique(),
        "Unique Sessions (Docs)": df['doc_id'].nunique(),
        "Missing Values (NaNs)": df.isna().sum().sum()
    }
    
    for k, v in stats.items():
        print(f"  - {k}: {v}")
        
    print("\nРаспределение действий (Action):")
    # 0 = DOWN, 1 = UP, 2 = MOVE
    action_counts = df['action'].value_counts(normalize=True) * 100
    for action, pct in action_counts.items():
        name = "DOWN (Нажатие)" if action == 0 else "UP (Отпускание)" if action == 1 else "MOVE (Движение)"
        print(f"  - {name}: {pct:.2f}%")

def extract_and_analyze_strokes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Экстракция полных свайпов (Strokes)...")
    
    # Очистка от NaN
    df = df.dropna(subset=['user_id', 'action', 'x', 'y', 'time_ms']).copy()
    df = df.sort_values(by=['user_id', 'doc_id', 'time_ms']).reset_index(drop=True)
    
    strokes = []
    
    for (user, doc), group in df.groupby(['user_id', 'doc_id']):
        downs = group[group['action'] == 0].index.tolist()
        ups = group[group['action'] == 1].index.tolist()
        
        for d_idx in downs:
            u_idxs = [u for u in ups if u > d_idx]
            if not u_idxs:
                continue
            u_idx = u_idxs[0]
            
            stroke = group.loc[d_idx:u_idx]
            if len(stroke) < 3: # Игнорируем короткие тапы без движения
                continue
                
            duration = stroke.iloc[-1]['time_ms'] - stroke.iloc[0]['time_ms']
            length_px = np.sqrt((stroke.iloc[-1]['x'] - stroke.iloc[0]['x'])**2 + 
                                (stroke.iloc[-1]['y'] - stroke.iloc[0]['y'])**2)
            
            strokes.append({
                'user_id': user,
                'doc_id': doc,
                'duration_ms': duration,
                'length_px': length_px,
                'velocity': length_px / duration if duration > 0 else 0,
                'median_pressure': stroke['pressure'].median(),
                'points_count': len(stroke)
            })
            
    strokes_df = pd.DataFrame(strokes)
    logger.info(f"Извлечено валидных свайпов: {len(strokes_df):,}")
    
    print("\nСтатистика по свайпам:")
    print(strokes_df[['duration_ms', 'length_px', 'velocity', 'median_pressure']].describe().round(2))
    
    return strokes_df, df

def generate_visualizations(strokes_df: pd.DataFrame, raw_df: pd.DataFrame):
    logger.info(f"Генерация графиков в {REPORT_DIR}...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # График 1: Распределение длительности свайпа
    plt.figure(figsize=(10, 6))
    # Отсекаем аномальные зависания (> 2 секунд) для красивого графика
    filtered_durations = strokes_df[strokes_df['duration_ms'] < 2000]
    sns.histplot(data=filtered_durations, x='duration_ms', bins=50, kde=True, color='blue')
    plt.title('Распределение длительности свайпов (мс)')
    plt.xlabel('Длительность (мс)')
    plt.ylabel('Количество')
    plt.savefig(REPORT_DIR / 'stroke_duration_dist.png', dpi=300)
    plt.close()

    # График 2: Траектории свайпов (WOW-эффект для презентации)
    # Берем двух случайных пользователей и рисуем их движения
    plt.figure(figsize=(10, 8))
    sample_users = raw_df['user_id'].unique()[:2]
    colors = ['red', 'blue']
    
    for i, user in enumerate(sample_users):
        user_data = raw_df[(raw_df['user_id'] == user) & (raw_df['action'] == 2)].head(2000)
        plt.scatter(user_data['x'], user_data['y'], s=2, alpha=0.5, color=colors[i], label=f'User {user}')
        
    plt.title('Сравнение микромоторики (Траектории движений 2 пользователей)')
    plt.xlabel('Координата X')
    plt.ylabel('Координата Y')
    plt.gca().invert_yaxis() # Инвертируем Y, так как координаты экрана идут сверху вниз
    plt.legend()
    plt.savefig(REPORT_DIR / 'user_trajectories_comparison.png', dpi=300)
    plt.close()
    
    logger.info("✅ Графики успешно сохранены!")

def run_eda():
    print("="*50)
    print("🚀 СТАРТ АНАЛИЗА ДАТАСЕТА TOUCHALYTICS")
    print("="*50)
    
    df = load_data()
    basic_statistics(df)
    strokes_df, df = extract_and_analyze_strokes(df)
    generate_visualizations(strokes_df, df)
    
    print("="*50)
    print(f"📁 Отчеты и графики лежат здесь: {REPORT_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_eda()