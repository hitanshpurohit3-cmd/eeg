import pandas as pd
from pathlib import Path

raw_file = Path("data/raw/participants.tsv")

df = pd.read_csv(raw_file, sep="\t")

print(df.head())
print("\nColumns:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)