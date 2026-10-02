import pandas as pd

events = pd.read_csv("data/raw/ufcstats/ufc_event_details.csv")
results = pd.read_csv("data/raw/ufcstats/ufc_fight_results.csv")

# Clean event names
events["EVENT"] = events["EVENT"].str.strip()
results["EVENT"] = results["EVENT"].str.strip()

# Parse dates
events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")

# Keep only event name + date
event_dates = events[["EVENT", "DATE"]].drop_duplicates()

# Check whether each fight can find an event date
merged = results.merge(
    event_dates,
    on="EVENT",
    how="left"
)

print("EVENT DATA")
print("Events:", len(events))
print("Unique events:", events["EVENT"].nunique())
print("Missing event dates:", events["DATE"].isna().sum())

print("\nFIGHT DATA")
print("Fight rows:", len(results))
print("Unique fights:", results[["EVENT", "BOUT"]].drop_duplicates().shape[0])

print("\nDATE MATCHING")
print("Fights with date:", merged["DATE"].notna().sum())
print("Fights without date:", merged["DATE"].isna().sum())

print("\nDATE RANGE")
print("Earliest:", merged["DATE"].min())
print("Latest:", merged["DATE"].max())

print("\nSAMPLE")
print(
    merged[
        ["EVENT", "BOUT", "DATE", "OUTCOME", "WEIGHTCLASS"]
    ].head(10).to_string(index=False)
)
