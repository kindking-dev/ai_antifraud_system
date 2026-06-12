import pandas as pd

df = pd.read_parquet("data/train_transaction.parquet.gzip")

print(df.shape)
print(df.columns[:20])