from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PARQUET_PATH = BASE_DIR / "data" / "train.parquet"


DTYPE_MAP = {
    "TransactionAmt": "float32",
    "isFraud": "int8",
    "TransactionDT": "float32",
}


def _read_csv_optimized(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"📦 Reading CSV: {path.name}")

    df = pd.read_csv(
        path,
        dtype=DTYPE_MAP,
        low_memory=True
    )

    print(f"✔ Loaded: {df.shape}")
    return df


def load_ieee_train() -> pd.DataFrame:
    """
    Smart loader:
    1. parquet (fast)
    2. fallback to CSV
    """

    # =========================
    # 1. TRY PARQUET
    # =========================
    if PARQUET_PATH.exists():
        print("⚡ Loading PARQUET (fast mode)")

        df = pd.read_parquet(PARQUET_PATH)

        print(f"✅ Loaded parquet: {df.shape}")
        return df

    # =========================
    # 2. FALLBACK CSV
    # =========================
    print("🐢 Parquet not found → using CSV")

    transaction_path = RAW_DATA_PATH / "train_transaction.csv"
    identity_path = RAW_DATA_PATH / "train_identity.csv"

    df_trans = _read_csv_optimized(transaction_path)
    df_id = _read_csv_optimized(identity_path)

    print("🔗 Merging datasets on TransactionID...")

    df = df_trans.merge(
        df_id,
        on="TransactionID",
        how="left"
    )

    print(f"✅ Final merged shape: {df.shape}")

    return df