import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Locate the dataset
# --------------------------------------------------

DATA_PATH = Path("data/raw/complete_ufc_data.csv")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 3. Basic dataset information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


# --------------------------------------------------
# 4. Column names
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")


# --------------------------------------------------
# 5. First five rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head().to_string())


# --------------------------------------------------
# 6. Data types
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes.to_string())


# --------------------------------------------------
# 7. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

missing = missing[missing["missing_count"] > 0]
missing = missing.sort_values(
    by="missing_count",
    ascending=False
)

if missing.empty:
    print("No missing values found.")
else:
    print(missing.to_string())


# --------------------------------------------------
# 8. Duplicate rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# --------------------------------------------------
# 9. Numerical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe().T.to_string())


# --------------------------------------------------
# 10. Categorical columns
# --------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    print(
        f"{column}: "
        f"{df[column].nunique(dropna=True)} unique values"
    )


# --------------------------------------------------
# 11. Save inspection report
# --------------------------------------------------

OUTPUT_DIR = Path("results/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report_path = OUTPUT_DIR / "dataset_profile.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("UFC DATASET PROFILE\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}\n\n"
    )

    file.write("COLUMNS\n")
    file.write("-" * 60 + "\n")

    for column in df.columns:
        file.write(f"{column}\n")

    file.write("\nDATA TYPES\n")
    file.write("-" * 60 + "\n")
    file.write(df.dtypes.to_string())

    file.write("\n\nMISSING VALUES\n")
    file.write("-" * 60 + "\n")
    file.write(missing.to_string())

    file.write("\n\nDUPLICATES\n")
    file.write("-" * 60 + "\n")
    file.write(f"Duplicate rows: {duplicate_count}\n")


print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
print(f"Report saved to: {report_path}")