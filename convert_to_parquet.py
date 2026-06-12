import pandas as pd
from pathlib import Path

BASE = Path("data/raw")
OUT = Path("data")

print("📦 Loading CSV...")

train_tx = pd.read_csv(BASE / "train_transaction.csv")
train_id = pd.read_csv(BASE / "train_identity.csv")

print(f"TX shape: {train_tx.shape}")
print(f"ID shape: {train_id.shape}")

print("🔗 Merging...")

df = train_tx.merge(train_id, on="TransactionID", how="left")

print(f"Final shape: {df.shape}")

print("💾 Saving parquet...")

OUT.mkdir(exist_ok=True)

df.to_parquet(OUT / "train.parquet", index=False)

print("✅ DONE → data/train.parquet")