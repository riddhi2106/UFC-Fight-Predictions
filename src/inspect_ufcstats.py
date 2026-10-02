import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/ufcstats")

files = [
    "ufc_fight_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv"
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(" -", column)

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())