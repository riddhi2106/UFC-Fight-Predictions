import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/ufcstats")

details = pd.read_csv(DATA_DIR / "ufc_fight_details.csv")
results = pd.read_csv(DATA_DIR / "ufc_fight_results.csv")
stats = pd.read_csv(DATA_DIR / "ufc_fight_stats.csv")

print("=" * 70)
print("1. DATASET SIZES")
print("=" * 70)

print("Details rows:", len(details))
print("Results rows:", len(results))
print("Stats rows:", len(stats))

print("\nUnique details URLs:", details["URL"].nunique())
print("Unique results URLs:", results["URL"].nunique())

print("\n" + "=" * 70)
print("2. STATS IDENTIFIERS")
print("=" * 70)

print("Unique stats EVENT+BOUT:",
      stats[["EVENT", "BOUT"]].drop_duplicates().shape[0])

print("Unique stats BOUT:",
      stats["BOUT"].nunique())

print("\n" + "=" * 70)
print("3. DUPLICATES")
print("=" * 70)

print("Duplicate detail URLs:",
      details["URL"].duplicated().sum())

print("Duplicate result URLs:",
      results["URL"].duplicated().sum())

print("Duplicate stats EVENT+BOUT:",
      stats.duplicated(["EVENT", "BOUT"]).sum())

print("\n" + "=" * 70)
print("4. EVENT+BOUT MATCH")
print("=" * 70)

results_key = results[["EVENT", "BOUT"]].copy()
stats_key = stats[["EVENT", "BOUT"]].copy()

results_key["EVENT"] = results_key["EVENT"].str.strip()
stats_key["EVENT"] = stats_key["EVENT"].str.strip()

results_key = results_key.drop_duplicates()
stats_key = stats_key.drop_duplicates()

merged = results_key.merge(
    stats_key,
    on=["EVENT", "BOUT"],
    how="outer",
    indicator=True
)

print(merged["_merge"].value_counts())

print("\n" + "=" * 70)
print("5. MATCHING FIGHTS")
print("=" * 70)

print(
    "Results with statistics:",
    (merged["_merge"] == "both").sum()
)

print(
    "Results without statistics:",
    (merged["_merge"] == "left_only").sum()
)

print(
    "Statistics without result:",
    (merged["_merge"] == "right_only").sum()
)

print("\n" + "=" * 70)
print("6. FIGHTERS PER FIGHT")
print("=" * 70)

fighter_counts = (
    stats.groupby(["EVENT", "BOUT"])["FIGHTER"]
    .nunique()
)

print(fighter_counts.value_counts().sort_index())

print("\nExactly two fighters:",
      (fighter_counts == 2).sum())

print("Not exactly two:",
      (fighter_counts != 2).sum())