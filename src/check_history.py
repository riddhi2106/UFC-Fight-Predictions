import pandas as pd

DATA_PATH = "data/raw/complete_ufc_data.csv"

df = pd.read_csv(DATA_PATH)

print("Total fights:", len(df))

print("\nUnique fighters:")
fighters = set(df["fighter1"].dropna()) | set(df["fighter2"].dropna())
print(len(fighters))

print("\nTop fighters by number of appearances:")
fighter_counts = pd.concat([
    df["fighter1"],
    df["fighter2"]
]).value_counts()

print(fighter_counts.head(20))

print("\nExample fighter history:")
fighter = fighter_counts.index[0]

fighter_history = df[
    (df["fighter1"] == fighter) |
    (df["fighter2"] == fighter)
].sort_values("event_date")

print(
    fighter_history[
        [
            "event_date",
            "fighter1",
            "fighter2",
            "outcome",
            "fighter1_sig_strikes_landed_pm",
            "fighter2_sig_strikes_landed_pm",
            "fighter1_takedown_avg_per15m",
            "fighter2_takedown_avg_per15m"
        ]
    ].to_string(index=False)
)