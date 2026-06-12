from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_PATH = BASE_DIR / "data" / "raw"
REPORT_PATH = BASE_DIR / "reports"

REPORT_PATH.mkdir(exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    print("📦 Loading raw CSV data...")

    df_trans = pd.read_csv(RAW_PATH / "train_transaction.csv", low_memory=False)
    df_id = pd.read_csv(RAW_PATH / "train_identity.csv", low_memory=False)

    df = df_trans.merge(df_id, on="TransactionID", how="left")

    print(f"✅ Loaded: {df.shape}")

    return df


# =========================
# BASIC ANALYSIS
# =========================

def basic_info(df):
    print("\n=== BASIC INFO ===")
    print(df.shape)
    print(df["isFraud"].value_counts(normalize=True))


# =========================
# MISSING ANALYSIS
# =========================

def missing_analysis(df):
    print("\n=== MISSING VALUES ===")

    missing = df.isnull().mean().sort_values(ascending=False)

    missing_df = missing.reset_index()
    missing_df.columns = ["feature", "missing_ratio"]

    missing_df.to_csv(REPORT_PATH / "missing_report.csv", index=False)

    print("Top missing:")
    print(missing.head(20))

    return missing


# =========================
# FRAUD DISTRIBUTION
# =========================

def fraud_analysis(df):
    print("\n=== FRAUD ANALYSIS ===")

    fraud_rate = df["isFraud"].mean()

    print(f"Fraud rate: {fraud_rate:.4f}")

    df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False).to_csv(
        REPORT_PATH / "fraud_by_product.csv"
    )

    df.groupby("card4")["isFraud"].mean().sort_values(ascending=False).to_csv(
        REPORT_PATH / "fraud_by_card4.csv"
    )


# =========================
# NUMERIC ANALYSIS
# =========================

def numeric_analysis(df):
    print("\n=== NUMERIC ANALYSIS ===")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    stats = df[num_cols].describe().T

    stats.to_csv(REPORT_PATH / "numeric_stats.csv")


# =========================
# CORRELATION WITH TARGET
# =========================

def correlation_analysis(df):
    print("\n=== CORRELATION ANALYSIS ===")

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns

    correlations = []

    for col in num_cols:
        if col != "isFraud":
            try:
                corr = df[col].corr(df["isFraud"])
                correlations.append((col, corr))
            except:  # noqa: E722
                continue

    corr_df = pd.DataFrame(correlations, columns=["feature", "corr"])
    corr_df = corr_df.sort_values(by="corr", ascending=False)

    corr_df.to_csv(REPORT_PATH / "correlation_with_target.csv", index=False)

    print(corr_df.head(20))


# =========================
# TIME ANALYSIS
# =========================

def time_analysis(df):
    print("\n=== TIME ANALYSIS ===")

    df = df.sort_values("TransactionDT")

    df["hour"] = (df["TransactionDT"] / 3600) % 24

    fraud_by_hour = df.groupby("hour")["isFraud"].mean()

    fraud_by_hour.to_csv(REPORT_PATH / "fraud_by_hour.csv")


# =========================
# MAIN
# =========================

def run_eda():
    df = load_data()

    basic_info(df)

    missing_analysis(df)

    fraud_analysis(df)

    numeric_analysis(df)

    correlation_analysis(df)

    time_analysis(df)

    print("\n✅ EDA COMPLETED")
    print(f"📁 Reports saved in: {REPORT_PATH}")


if __name__ == "__main__":
    run_eda()