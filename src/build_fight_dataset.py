import pandas as pd
from pathlib import Path


RAW = Path("data/raw/ufcstats")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def parse_landed_attempted(value):
    """
    Convert values such as:
        '29 of 73' -> (29, 73)

    Missing or malformed values become (0, 0).
    """
    if pd.isna(value):
        return 0.0, 0.0

    value = str(value).strip()

    if "of" not in value:
        return 0.0, 0.0

    parts = value.split("of")

    if len(parts) != 2:
        return 0.0, 0.0

    try:
        landed = float(parts[0].strip())
        attempted = float(parts[1].strip())
        return landed, attempted
    except ValueError:
        return 0.0, 0.0


def parse_number(value):
    """
    Convert a simple numeric value into float.
    """
    if pd.isna(value):
        return 0.0

    value = str(value).strip()

    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_control(value):
    """
    Convert control time '3:24' into seconds.
    """
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if ":" not in value:
        return 0

    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return 0


# =========================================================
# LOAD DATA
# =========================================================

print("Loading data...")

events = pd.read_csv(
    RAW / "ufc_event_details.csv"
)

results = pd.read_csv(
    RAW / "ufc_fight_results.csv"
)

stats = pd.read_csv(
    RAW / "ufc_fight_stats.csv"
)


# =========================================================
# CLEAN IDENTIFIERS
# =========================================================

events["EVENT"] = (
    events["EVENT"]
    .astype(str)
    .str.strip()
)

results["EVENT"] = (
    results["EVENT"]
    .astype(str)
    .str.strip()
)

results["BOUT"] = (
    results["BOUT"]
    .astype(str)
    .str.strip()
)

stats["EVENT"] = (
    stats["EVENT"]
    .astype(str)
    .str.strip()
)

stats["BOUT"] = (
    stats["BOUT"]
    .astype(str)
    .str.strip()
)

stats["FIGHTER"] = (
    stats["FIGHTER"]
    .astype(str)
    .str.strip()
)


# =========================================================
# EVENT DATES
# =========================================================

events["DATE"] = pd.to_datetime(
    events["DATE"],
    errors="coerce"
)

event_dates = (
    events[["EVENT", "DATE"]]
    .drop_duplicates(subset=["EVENT"])
)


# =========================================================
# HANDLE KNOWN EVENT-NAME MISMATCHES
# =========================================================

event_name_mapping = {
    "UFC Fight Night: Lopes vs. Silva":
        "Noche UFC: Lopes vs. Silva",

    "UFC Fight Night: Grasso vs. Shevchenko 2":
        "Noche UFC: Grasso vs. Shevchenko 2",
}


results["EVENT_FOR_DATE"] = (
    results["EVENT"]
    .replace(event_name_mapping)
)


results = results.merge(
    event_dates.rename(
        columns={"EVENT": "EVENT_FOR_DATE"}
    ),
    on="EVENT_FOR_DATE",
    how="left"
)


# =========================================================
# REMOVE DUPLICATE RESULT ROWS
# =========================================================

results = results.drop_duplicates(
    subset=["EVENT", "BOUT"],
    keep="first"
)


# =========================================================
# PARSE ROUND-LEVEL STATISTICS
# =========================================================

# These columns contain values like:
# "29 of 73"

landed_attempted_columns = [
    "KD",
    "SIG.STR.",
    "TOTAL STR.",
    "TD",
    "HEAD",
    "BODY",
    "LEG",
    "DISTANCE",
    "CLINCH",
    "GROUND",
]


for column in landed_attempted_columns:

    landed = []
    attempted = []

    for value in stats[column]:

        l, a = parse_landed_attempted(value)

        landed.append(l)
        attempted.append(a)

    clean_name = (
        column
        .replace(".", "")
        .replace(" ", "_")
    )

    stats[f"{clean_name}_LANDED"] = landed
    stats[f"{clean_name}_ATTEMPTED"] = attempted


# These are single numeric counts.

stats["SUB_ATT_LANDED"] = (
    stats["SUB.ATT"]
    .apply(parse_number)
)

stats["REV_LANDED"] = (
    stats["REV."]
    .apply(parse_number)
)


# Control time.

stats["CTRL_SECONDS"] = (
    stats["CTRL"]
    .apply(parse_control)
)


# =========================================================
# FIGHT-LEVEL AGGREGATION
# =========================================================

aggregation_columns = [
    "KD_LANDED",
    "KD_ATTEMPTED",

    "SIGSTR_LANDED",
    "SIGSTR_ATTEMPTED",

    "TOTAL_STR_LANDED",
    "TOTAL_STR_ATTEMPTED",

    "TD_LANDED",
    "TD_ATTEMPTED",

    "SUB_ATT_LANDED",

    "REV_LANDED",

    "HEAD_LANDED",
    "HEAD_ATTEMPTED",

    "BODY_LANDED",
    "BODY_ATTEMPTED",

    "LEG_LANDED",
    "LEG_ATTEMPTED",

    "DISTANCE_LANDED",
    "DISTANCE_ATTEMPTED",

    "CLINCH_LANDED",
    "CLINCH_ATTEMPTED",

    "GROUND_LANDED",
    "GROUND_ATTEMPTED",

    "CTRL_SECONDS",
]


fight_stats = (
    stats
    .groupby(
        ["EVENT", "BOUT", "FIGHTER"],
        as_index=False
    )[aggregation_columns]
    .sum()
)


# =========================================================
# KEEP FIGHTS WITH EXACTLY TWO FIGHTERS
# =========================================================

fighter_counts = (
    fight_stats
    .groupby(
        ["EVENT", "BOUT"]
    )["FIGHTER"]
    .nunique()
)


valid_fights = fighter_counts[
    fighter_counts == 2
].index


fight_stats = (
    fight_stats
    .set_index(
        ["EVENT", "BOUT"]
    )
    .loc[valid_fights]
    .reset_index()
)


# =========================================================
# CREATE FIGHTER A / FIGHTER B
# =========================================================

fight_stats["fighter_number"] = (
    fight_stats
    .groupby(
        ["EVENT", "BOUT"]
    )
    .cumcount()
)


fighter_a = (
    fight_stats[
        fight_stats["fighter_number"] == 0
    ]
    .drop(columns=["fighter_number"])
    .copy()
)


fighter_b = (
    fight_stats[
        fight_stats["fighter_number"] == 1
    ]
    .drop(columns=["fighter_number"])
    .copy()
)


# =========================================================
# RENAME FIGHTER A
# =========================================================

rename_a = {
    "FIGHTER": "fighter_a"
}

for column in aggregation_columns:

    rename_a[column] = (
        "a_"
        + column.lower()
    )


fighter_a = fighter_a.rename(
    columns=rename_a
)


# =========================================================
# RENAME FIGHTER B
# =========================================================

rename_b = {
    "FIGHTER": "fighter_b"
}

for column in aggregation_columns:

    rename_b[column] = (
        "b_"
        + column.lower()
    )


fighter_b = fighter_b.rename(
    columns=rename_b
)


# =========================================================
# MERGE FIGHTERS
# =========================================================

fight_level = fighter_a.merge(
    fighter_b,
    on=["EVENT", "BOUT"],
    how="inner"
)


# =========================================================
# ADD FIGHT INFORMATION
# =========================================================

result_columns = [
    "EVENT",
    "BOUT",
    "DATE",
    "OUTCOME",
    "WEIGHTCLASS",
    "METHOD",
    "ROUND",
    "TIME",
    "TIME FORMAT",
]


fight_level = fight_level.merge(
    results[result_columns],
    on=["EVENT", "BOUT"],
    how="left"
)


# =========================================================
# REMOVE FIGHTS WITHOUT DATES
# =========================================================

before_dates = len(fight_level)

fight_level = fight_level[
    fight_level["DATE"].notna()
].copy()

removed_dates = (
    before_dates - len(fight_level)
)

print(
    f"Removed {removed_dates} fights without dates."
)


# =========================================================
# REMOVE DRAWS AND NO-CONTESTS
# =========================================================

before_outcomes = len(fight_level)

fight_level = fight_level[
    fight_level["OUTCOME"].isin(
        ["W/L", "L/W"]
    )
].copy()

removed_outcomes = (
    before_outcomes - len(fight_level)
)

print(
    f"Removed {removed_outcomes} "
    "draw/no-contest fights."
)


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================

fight_level = (
    fight_level
    .sort_values(
        ["DATE", "EVENT", "BOUT"]
    )
    .reset_index(drop=True)
)


# =========================================================
# SAVE
# =========================================================

output_path = (
    OUT / "fight_level_data.csv"
)

fight_level.to_csv(
    output_path,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("FIGHT-LEVEL DATASET")
print("-------------------")

print(
    "Rows:",
    len(fight_level)
)

print(
    "Columns:",
    len(fight_level.columns)
)

print()
print("Date range:")

print(
    "Start:",
    fight_level["DATE"].min()
)

print(
    "End:",
    fight_level["DATE"].max()
)

print()
print("Outcome distribution:")

print(
    fight_level["OUTCOME"]
    .value_counts()
)

print()
print("Missing values:")

print(
    fight_level
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

print()
print("Saved to:")

print(output_path)