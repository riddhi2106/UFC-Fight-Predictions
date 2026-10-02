import pandas as pd
from pathlib import Path
from collections import defaultdict, deque


INPUT = Path("data/processed/fight_level_data.csv")
OUTPUT = Path("data/processed/ml_dataset.csv")


RECENT_WINDOWS = [3, 5]

DEFAULT_STRIKE_ACCURACY = 0.0
DEFAULT_TD_ACCURACY = 0.0
DEFAULT_TD_DEFENCE = 0.0


# =========================================================
# HELPERS
# =========================================================

def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_career_features(history):

    fights = history["fights"]
    total_fights = len(fights)

    if total_fights == 0:

        return {
            "total_fights": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,

            "sig_str_landed_per_fight": 0.0,
            "sig_str_attempted_per_fight": 0.0,
            "sig_str_accuracy": DEFAULT_STRIKE_ACCURACY,

            "td_landed_per_fight": 0.0,
            "td_attempted_per_fight": 0.0,
            "td_accuracy": DEFAULT_TD_ACCURACY,

            "sub_attempts_per_fight": 0.0,
            "reversals_per_fight": 0.0,

            "control_seconds_per_fight": 0.0,

            "sig_str_absorbed_per_fight": 0.0,
            "td_faced_per_fight": 0.0,
            "td_defence": DEFAULT_TD_DEFENCE,
        }

    wins = sum(f["win"] for f in fights)
    losses = total_fights - wins

    sig_landed = sum(
        f["sig_landed"] for f in fights
    )

    sig_attempted = sum(
        f["sig_attempted"] for f in fights
    )

    td_landed = sum(
        f["td_landed"] for f in fights
    )

    td_attempted = sum(
        f["td_attempted"] for f in fights
    )

    sub_attempts = sum(
        f["sub_attempts"] for f in fights
    )

    reversals = sum(
        f["reversals"] for f in fights
    )

    control_seconds = sum(
        f["control_seconds"] for f in fights
    )

    sig_absorbed = sum(
        f["sig_absorbed"] for f in fights
    )

    td_faced = sum(
        f["td_faced"] for f in fights
    )

    return {

        "total_fights": total_fights,

        "wins": wins,

        "losses": losses,

        "win_rate": safe_divide(
            wins,
            total_fights
        ),

        "sig_str_landed_per_fight": safe_divide(
            sig_landed,
            total_fights
        ),

        "sig_str_attempted_per_fight": safe_divide(
            sig_attempted,
            total_fights
        ),

        "sig_str_accuracy": safe_divide(
            sig_landed,
            sig_attempted
        ),

        "td_landed_per_fight": safe_divide(
            td_landed,
            total_fights
        ),

        "td_attempted_per_fight": safe_divide(
            td_attempted,
            total_fights
        ),

        "td_accuracy": safe_divide(
            td_landed,
            td_attempted
        ),

        "sub_attempts_per_fight": safe_divide(
            sub_attempts,
            total_fights
        ),

        "reversals_per_fight": safe_divide(
            reversals,
            total_fights
        ),

        "control_seconds_per_fight": safe_divide(
            control_seconds,
            total_fights
        ),

        "sig_str_absorbed_per_fight": safe_divide(
            sig_absorbed,
            total_fights
        ),

        "td_faced_per_fight": safe_divide(
            td_faced,
            total_fights
        ),

        "td_defence": (
            1.0 - safe_divide(
                td_landed,
                td_faced
            )
            if td_faced > 0
            else DEFAULT_TD_DEFENCE
        ),
    }


def get_recent_features(history):

    recent = history["recent"]

    features = {}

    for window in RECENT_WINDOWS:

        fights = list(recent)[-window:]

        if len(fights) == 0:

            features[
                f"last_{window}_win_rate"
            ] = 0.0

            features[
                f"last_{window}_sig_str_accuracy"
            ] = DEFAULT_STRIKE_ACCURACY

            continue

        wins = sum(
            f["win"]
            for f in fights
        )

        sig_landed = sum(
            f["sig_landed"]
            for f in fights
        )

        sig_attempted = sum(
            f["sig_attempted"]
            for f in fights
        )

        features[
            f"last_{window}_win_rate"
        ] = safe_divide(
            wins,
            len(fights)
        )

        features[
            f"last_{window}_sig_str_accuracy"
        ] = safe_divide(
            sig_landed,
            sig_attempted
        )

    return features


def get_fighter_features(history):

    features = {}

    features.update(
        get_career_features(history)
    )

    features.update(
        get_recent_features(history)
    )

    return features


# =========================================================
# LOAD DATA
# =========================================================

print("Loading fight-level data...")

df = pd.read_csv(INPUT)

df["DATE"] = pd.to_datetime(
    df["DATE"]
)

df = (
    df
    .sort_values(
        ["DATE", "EVENT"]
    )
    .reset_index(drop=True)
)


# =========================================================
# HISTORY
# =========================================================

histories = defaultdict(
    lambda: {
        "fights": [],
        "recent": deque(maxlen=5)
    }
)


feature_names = [

    "total_fights",
    "wins",
    "losses",
    "win_rate",

    "sig_str_landed_per_fight",
    "sig_str_attempted_per_fight",
    "sig_str_accuracy",

    "td_landed_per_fight",
    "td_attempted_per_fight",
    "td_accuracy",

    "sub_attempts_per_fight",
    "reversals_per_fight",

    "control_seconds_per_fight",

    "sig_str_absorbed_per_fight",
    "td_faced_per_fight",
    "td_defence",

    "last_3_win_rate",
    "last_3_sig_str_accuracy",

    "last_5_win_rate",
    "last_5_sig_str_accuracy",
]


# =========================================================
# CHRONOLOGICAL EVENT PROCESSING
# =========================================================

feature_rows = []

print(
    "Building event-level chronological histories..."
)


for (date, event), event_df in df.groupby(
    ["DATE", "EVENT"],
    sort=True
):

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # SNAPSHOT HISTORIES BEFORE ANY FIGHT IN THIS EVENT.
    #
    # Therefore no fight from this event can leak into
    # another fight from the same event.
    # -----------------------------------------------------

    event_feature_rows = []

    for _, row in event_df.iterrows():

        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]

        history_a = histories[fighter_a]
        history_b = histories[fighter_b]

        features_a = get_fighter_features(
            history_a
        )

        features_b = get_fighter_features(
            history_b
        )

        row_features = {

            "DATE": row["DATE"],
            "EVENT": row["EVENT"],
            "BOUT": row["BOUT"],

            "fighter_a": fighter_a,
            "fighter_b": fighter_b,

            "WEIGHTCLASS": row["WEIGHTCLASS"],

            "target": (
                1
                if row["OUTCOME"] == "W/L"
                else 0
            ),
        }

        for feature in feature_names:

            row_features[
                f"a_{feature}"
            ] = features_a[feature]

            row_features[
                f"b_{feature}"
            ] = features_b[feature]

            row_features[
                f"diff_{feature}"
            ] = (
                features_a[feature]
                -
                features_b[feature]
            )

        event_feature_rows.append(
            row_features
        )


    # -----------------------------------------------------
    # ADD FEATURES ONLY AFTER ALL FIGHTS IN EVENT HAVE
    # BEEN REPRESENTED.
    # -----------------------------------------------------

    feature_rows.extend(
        event_feature_rows
    )


    # -----------------------------------------------------
    # NOW UPDATE ALL FIGHTER HISTORIES USING THIS EVENT.
    # -----------------------------------------------------

    for _, row in event_df.iterrows():

        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]

        a_won = (
            row["OUTCOME"] == "W/L"
        )

        b_won = (
            row["OUTCOME"] == "L/W"
        )


        a_fight_record = {

            "win": int(a_won),

            "sig_landed": row[
                "a_sigstr_landed"
            ],

            "sig_attempted": row[
                "a_sigstr_attempted"
            ],

            "td_landed": row[
                "a_td_landed"
            ],

            "td_attempted": row[
                "a_td_attempted"
            ],

            "sub_attempts": row[
                "a_sub_att_landed"
            ],

            "reversals": row[
                "a_rev_landed"
            ],

            "control_seconds": row[
                "a_ctrl_seconds"
            ],

            "sig_absorbed": row[
                "b_sigstr_landed"
            ],

            "td_faced": row[
                "b_td_attempted"
            ],
        }


        b_fight_record = {

            "win": int(b_won),

            "sig_landed": row[
                "b_sigstr_landed"
            ],

            "sig_attempted": row[
                "b_sigstr_attempted"
            ],

            "td_landed": row[
                "b_td_landed"
            ],

            "td_attempted": row[
                "b_td_attempted"
            ],

            "sub_attempts": row[
                "b_sub_att_landed"
            ],

            "reversals": row[
                "b_rev_landed"
            ],

            "control_seconds": row[
                "b_ctrl_seconds"
            ],

            "sig_absorbed": row[
                "a_sigstr_landed"
            ],

            "td_faced": row[
                "a_td_attempted"
            ],
        }


        histories[
            fighter_a
        ]["fights"].append(
            a_fight_record
        )

        histories[
            fighter_a
        ]["recent"].append(
            a_fight_record
        )


        histories[
            fighter_b
        ]["fights"].append(
            b_fight_record
        )

        histories[
            fighter_b
        ]["recent"].append(
            b_fight_record
        )


# =========================================================
# CREATE DATAFRAME
# =========================================================

ml_df = pd.DataFrame(
    feature_rows
)


# =========================================================
# SANITY CHECKS
# =========================================================

print()
print("ML DATASET")
print("----------")

print(
    "Rows:",
    len(ml_df)
)

print(
    "Columns:",
    len(ml_df.columns)
)

print()
print("Target distribution:")

print(
    ml_df["target"].value_counts()
)

print()
print("Missing values:")

print(
    ml_df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


# =========================================================
# SAVE
# =========================================================

ml_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved to:")

print(OUTPUT)