import pandas as pd
import numpy as np
from pathlib import Path
import structlog
import random

# Настройка логирования
logging_config = structlog.get_logger(__name__)
logger = logging_config.bind(stage="FeatureBuilder")

# =========================
# CONFIG & HYPERPARAMETERS
# =========================
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = Path(r"C:\Users\Islam\Desktop\Diploma\ai_antifraud_system\data\raw\raw_touches.csv")
OUT_DIR = BASE_DIR / "data" / "processed"

# Архитектурные параметры
WINDOW_SIZE = 15  # Количество свайпов для скоринга
TRAIN_DOCS = [1, 2, 3, 4, 5]  # Сессии для создания профиля-эталона
TEST_DOCS = [6, 7]            # Сессии для имитации атак и инференса

# Фильтры аномалий (найденные на EDA)
MIN_DURATION, MAX_DURATION = 10, 3000  # миллисекунды
MIN_LENGTH = 10                        # пиксели
MAX_VELOCITY = 20                      # пикс/мс

def load_and_clean_data(filepath: Path) -> pd.DataFrame:
    logger.info("Загрузка и первичная очистка данных...")
    cols = ['phone_id', 'user_id', 'doc_id', 'time_ms', 'action', 
            'phone_orient', 'x', 'y', 'pressure', 'area', 'finger_orient']
    
    df = pd.read_csv(filepath, names=cols, header=0, on_bad_lines='skip')
    df = df.dropna(subset=['user_id', 'action', 'x', 'y', 'time_ms'])
    df = df.sort_values(by=['user_id', 'doc_id', 'time_ms']).reset_index(drop=True)
    return df

def extract_clean_strokes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Экстракция свайпов и фильтрация выбросов (Data Cleansing)...")
    strokes = []
    
    for (user, doc), group in df.groupby(['user_id', 'doc_id']):
        downs = group[group['action'] == 0].index.tolist()
        ups = group[group['action'] == 1].index.tolist()
        
        for d_idx in downs:
            u_idxs = [u for u in ups if u > d_idx]
            if not u_idxs:
                continue
            u_idx = u_idxs[0]
            
            stroke_data = group.loc[d_idx:u_idx]
            if len(stroke_data) < 3:
                continue
                
            duration = stroke_data.iloc[-1]['time_ms'] - stroke_data.iloc[0]['time_ms']
            length_x = stroke_data.iloc[-1]['x'] - stroke_data.iloc[0]['x']
            length_y = stroke_data.iloc[-1]['y'] - stroke_data.iloc[0]['y']
            trajectory_length = np.sqrt(length_x**2 + length_y**2)
            velocity = trajectory_length / duration if duration > 0 else 0
            
            # Применяем фильтры из EDA
            if not (MIN_DURATION <= duration <= MAX_DURATION): continue  # noqa: E701
            if trajectory_length < MIN_LENGTH: continue  # noqa: E701
            if velocity > MAX_VELOCITY: continue  # noqa: E701
            
            strokes.append({
                'user_id': user,
                'doc_id': doc,
                'duration_ms': duration,
                'length_px': trajectory_length,
                'velocity': velocity,
                'median_pressure': stroke_data['pressure'].median(),
                'median_area': stroke_data['area'].median(),
            })
            
    strokes_df = pd.DataFrame(strokes)
    logger.info(f"Осталось валидных, чистых свайпов: {len(strokes_df)}")
    return strokes_df

def build_windows(strokes_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Сборка микромоторики в окна (N={WINDOW_SIZE})...")
    windows = []
    
    for (user, doc), group in strokes_df.groupby(['user_id', 'doc_id']):
        group = group.reset_index(drop=True)
        # Tumbling windows
        for i in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window_slice = group.iloc[i:i+WINDOW_SIZE]
            
            w_features = {'user_id': user, 'doc_id': doc}
            
            for col in ['duration_ms', 'length_px', 'velocity', 'median_pressure', 'median_area']:
                w_features[f'{col}_mean'] = window_slice[col].mean()
                w_features[f'{col}_std'] = window_slice[col].std()
                w_features[f'{col}_max'] = window_slice[col].max()
                
            windows.append(w_features)
            
    return pd.DataFrame(windows).fillna(0)

def generate_contrastive_dataset(windows_df: pd.DataFrame):
    logger.info("Генерация расширенных пар 'Окно-Профиль' (Temporal Split)...")
    
    feature_cols = [c for c in windows_df.columns if c not in ['user_id', 'doc_id']]
    dataset = []
    profiles_list = []
    users = windows_df['user_id'].unique()
    
    for user in users:
        user_windows = windows_df[windows_df['user_id'] == user].reset_index(drop=True)
        if len(user_windows) < 5:
            continue # Пропускаем юзеров, у которых почти нет данных
            
        # Берем первые 40% окон по времени как эталон (Профиль)
        split_idx = max(1, int(len(user_windows) * 0.4))
        profile_windows = user_windows.iloc[:split_idx]
        test_windows = user_windows.iloc[split_idx:]
        
        # Считаем Профиль
        profile = profile_windows[feature_cols].median()
        profiles_list.append({'user_id': user, **{f'prof_{k}': v for k, v in profile.items()}})
        
        # Генерируем обучающие сэмплы
        impostors = [u for u in users if u != user]
        
        for _, window in test_windows.iterrows():
            # Класс 1 (Владелец): Окно владельца минус его профиль
            pos_sample = {col: abs(window[col] - profile[col]) for col in feature_cols}
            pos_sample['target'] = 1
            dataset.append(pos_sample)
            
            # Класс 0 (Атака): Окно 2-х случайных мошенников минус профиль владельца
            for _ in range(2):
                impostor_id = random.choice(impostors)
                impostor_window = windows_df[windows_df['user_id'] == impostor_id].sample(1).iloc[0]
                neg_sample = {col: abs(impostor_window[col] - profile[col]) for col in feature_cols}
                neg_sample['target'] = 0
                dataset.append(neg_sample)
                
    final_df = pd.DataFrame(dataset)
    profiles_df = pd.DataFrame(profiles_list)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_DIR / "touchalytics_contrastive.parquet", index=False)
    profiles_df.to_parquet(OUT_DIR / "user_profiles.parquet", index=False)
    
    logger.info(f"✅ Расширенный датасет готов | Форма: {final_df.shape}")
    logger.info(f"Баланс таргетов:\n{final_df['target'].value_counts()}")

if __name__ == "__main__":
    df = load_and_clean_data(RAW_DATA_PATH)
    strokes = extract_clean_strokes(df)
    windows = build_windows(strokes)
    generate_contrastive_dataset(windows)