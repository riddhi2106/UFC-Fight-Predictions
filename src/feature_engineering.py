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

            # a_won is resolved by name in
            # build_fight_dataset.py. OUTCOME alone is
            # stated relative to the BOUT string and does
            # not line up with fighter_a.
            "target": int(row["a_won"]),
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

        a_won = bool(row["a_won"])

        b_won = not a_won


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
# PHYSICAL ATTRIBUTE FEATURES
# =========================================================
#
# Height, reach, stance and date of birth are static
# fighter attributes. They do not change from fight to
# fight, so joining them introduces no temporal leakage.
# Age is derived per fight from the event date, so it is
# always the age the fighter actually was on fight night.
#
# The reference CS229 study identified age difference as
# one of its most important variables, and none of the
# history features above capture physical attributes.

PHYSICAL_SOURCE = Path(
    "data/raw/complete_ufc_data.csv"
)

physical_raw = pd.read_csv(
    PHYSICAL_SOURCE
)


# Fighters appear in both the fighter1 and fighter2
# columns, so stack the two halves into one lookup.

lookup_parts = []

for side in ["1", "2"]:

    lookup_parts.append(
        physical_raw[
            [
                f"fighter{side}",
                f"fighter{side}_height",
                f"fighter{side}_reach",
                f"fighter{side}_dob",
                f"fighter{side}_stance",
            ]
        ].rename(
            columns={
                f"fighter{side}": "fighter",
                f"fighter{side}_height": "height",
                f"fighter{side}_reach": "reach",
                f"fighter{side}_dob": "dob",
                f"fighter{side}_stance": "stance",
            }
        )
    )


physical = (
    pd.concat(
        lookup_parts,
        ignore_index=True
    )
    .dropna(subset=["fighter"])
)

physical["fighter"] = (
    physical["fighter"].str.strip()
)

physical = physical.drop_duplicates(
    subset=["fighter"]
).set_index("fighter")

physical["dob"] = pd.to_datetime(
    physical["dob"],
    errors="coerce"
)

physical["is_southpaw"] = (
    physical["stance"]
    .eq("Southpaw")
    .astype(float)
)

physical.loc[
    physical["stance"].isna(),
    "is_southpaw"
] = float("nan")


ml_df["DATE"] = pd.to_datetime(
    ml_df["DATE"]
)


def attach_side(frame, side):
    """Map one side's physical attributes onto the rows."""

    names = frame[f"fighter_{side}"].str.strip()

    height = names.map(physical["height"])
    reach = names.map(physical["reach"])
    dob = names.map(physical["dob"])
    southpaw = names.map(physical["is_southpaw"])

    age = (
        (frame["DATE"] - dob).dt.days
        / 365.25
    )

    # Guard against obviously corrupt dates of birth.
    age = age.where(
        (age > 15) & (age < 60)
    )

    return height, reach, age, southpaw


(
    a_height,
    a_reach,
    a_age,
    a_southpaw,
) = attach_side(ml_df, "a")

(
    b_height,
    b_reach,
    b_age,
    b_southpaw,
) = attach_side(ml_df, "b")


physical_pairs = {
    "age": (a_age, b_age),
    "height": (a_height, b_height),
    "reach": (a_reach, b_reach),
    "southpaw": (a_southpaw, b_southpaw),
}


for name, (a_values, b_values) in physical_pairs.items():

    ml_df[f"a_{name}"] = a_values
    ml_df[f"b_{name}"] = b_values

    difference = a_values - b_values

    # A known flag is required because 0.0 is both the
    # fill value and a genuine difference. Without it the
    # model cannot tell "no edge" from "unknown".
    ml_df[f"diff_{name}_known"] = (
        difference.notna().astype(int)
    )

    ml_df[f"diff_{name}"] = difference.fillna(0.0)


# Orthodox vs southpaw is a known stylistic matchup, and
# it is symmetric, so it is kept as its own flag rather
# than as a difference.

ml_df["diff_stance_mismatch"] = (
    (a_southpaw != b_southpaw)
    & a_southpaw.notna()
    & b_southpaw.notna()
).astype(int)


print()
print("PHYSICAL FEATURE COVERAGE")
print("-------------------------")

for name in physical_pairs:

    print(
        f"{name:<10}",
        f"{ml_df[f'diff_{name}_known'].mean():.4f}"
    )


# =========================================================
# FIGHTER PROFILES
# =========================================================
#
# After the loop above, `histories` holds each fighter's
# full career state. Dumping it lets src/predict.py score
# a matchup that is not in the dataset without rebuilding
# the whole pipeline.
#
# These are end-of-dataset values, so they are only valid
# for predicting a FUTURE fight, never for re-scoring a
# past one.

PROFILE_OUTPUT = Path(
    "data/processed/fighter_profiles.csv"
)

profile_rows = []

for fighter_name, history in histories.items():

    profile = {"fighter": fighter_name}

    profile.update(
        get_fighter_features(history)
    )

    profile_rows.append(profile)


profiles = pd.DataFrame(profile_rows)

# Carry the static physical attributes across so predict.py
# needs one file rather than two.
profiles = profiles.join(
    physical[["height", "reach", "dob", "is_southpaw"]],
    on="fighter",
)

profiles = profiles.sort_values(
    "total_fights",
    ascending=False,
)

profiles.to_csv(
    PROFILE_OUTPUT,
    index=False,
)

print()
print(
    f"Saved {len(profiles)} fighter profiles to: "
    f"{PROFILE_OUTPUT}"
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