"""
Exploratory Data Analysis (EDA) Baseline Script.
Calculates class imbalance and missing values for the IEEE-CIS dataset.
"""

import os
import sys

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

def analyze_dataset(file_path: str) -> None:
    if not os.path.exists(file_path):
        logger.error("dataset_not_found", path=file_path)
        sys.exit(1)

    logger.info("loading_parquet_dataset", path=file_path)
    
    try:
        # Считывание данных за O(1) по колонкам через pyarrow
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        target_col = 'isFraud'
        if target_col not in df.columns:
            logger.error("target_column_missing", column=target_col)
            sys.exit(1)

        # 1. Расчет дисбаланса классов
        total_rows = len(df)
        fraud_counts = df[target_col].value_counts(normalize=True) * 100
        
        fraud_ratio = fraud_counts.get(1, 0.0)
        normal_ratio = fraud_counts.get(0, 0.0)

        # 2. Анализ пропусков (NaN)
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / total_rows) * 100
        top_missing = missing_percent[missing_percent > 0].sort_values(ascending=False).head(10)

        # 3. Вывод метрик
        print("\n" + "="*40)
        print("📊 DATASET BASELINE ANALYSIS")
        print("="*40)
        print(f"Total Transactions: {total_rows:,}")
        print(f"Normal (0): {normal_ratio:.3f}%")
        print(f"Fraud (1):  {fraud_ratio:.3f}%")
        print("-" * 40)
        print("Top 10 Columns with Missing Data (%):")
        print(top_missing.to_string())
        print("="*40 + "\n")
        
        logger.info(
            "eda_completed_successfully", 
            total_rows=total_rows, 
            fraud_ratio_pct=round(fraud_ratio, 3)
        )

    except Exception as e:
        logger.exception("eda_pipeline_failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    # Относительный путь от корня проекта
    DATASET_PATH = "data/train_transaction.parquet.gzip"
    analyze_dataset(DATASET_PATH)