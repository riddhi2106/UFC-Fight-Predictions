import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque


INPUT = Path("data/processed/fight_level_data.csv")
OUTPUT = Path("data/processed/ml_dataset.csv")


RECENT_WINDOWS = [3, 5]

DEFAULT_STRIKE_ACCURACY = 0.0
DEFAULT_STRIKE_DEFENCE = 0.0
DEFAULT_TD_ACCURACY = 0.0
DEFAULT_TD_DEFENCE = 0.0

DEFAULT_ELO = 1500.0
INITIAL_ELO_K = 64.0
BASE_ELO_K = 32.0

HALF_LIFE_DAYS = 730.0  # 2-year half-life for exponential recency decay
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS


# =========================================================
# HELPERS
# =========================================================

def safe_divide(numerator, denominator):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_career_features(history, current_date=None):

    fights = history["fights"]
    total_fights = len(fights)
    elo_val = history.get("elo", DEFAULT_ELO)

    # Layoff duration and UFC debut factor
    if history.get("last_fight_date") is None or current_date is None:
        days_since = 365.0
        is_debut = 1.0 if total_fights == 0 else 0.0
    else:
        days_since = min(max(0.0, float((current_date - history["last_fight_date"]).days)), 1000.0)
        is_debut = 0.0

    if total_fights == 0:

        return {
            "total_fights": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "elo": elo_val,
            "is_debut": is_debut,
            "days_since_last_fight": days_since,
            "decayed_win_rate": 0.0,

            "sig_str_landed_per_fight": 0.0,
            "sig_str_attempted_per_fight": 0.0,
            "sig_str_accuracy": DEFAULT_STRIKE_ACCURACY,
            "sig_str_defence": DEFAULT_STRIKE_DEFENCE,
            "decayed_sig_str_defence": DEFAULT_STRIKE_DEFENCE,

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

    sig_faced = sum(
        f.get("sig_faced", 0.0) for f in fights
    )

    td_faced = sum(
        f["td_faced"] for f in fights
    )

    td_absorbed = sum(
        f.get("td_absorbed", 0.0) for f in fights
    )

    # Exponential time-decay weights
    if current_date is not None:
        weights = [np.exp(-DECAY_RATE * max(0.0, float((current_date - f["date"]).days))) for f in fights]
    else:
        weights = [1.0] * total_fights

    w_sum = sum(weights)
    decayed_win_rate = sum(w * f["win"] for w, f in zip(weights, fights)) / w_sum if w_sum > 0 else 0.0

    w_sig_absorbed = sum(w * f["sig_absorbed"] for w, f in zip(weights, fights))
    w_sig_faced = sum(w * f.get("sig_faced", 0.0) for w, f in zip(weights, fights))
    decayed_sig_defence = (
        max(0.0, min(1.0, 1.0 - safe_divide(w_sig_absorbed, w_sig_faced)))
        if w_sig_faced > 0
        else DEFAULT_STRIKE_DEFENCE
    )

    return {

        "total_fights": total_fights,

        "wins": wins,

        "losses": losses,

        "win_rate": safe_divide(
            wins,
            total_fights
        ),

        "elo": elo_val,
        "is_debut": is_debut,
        "days_since_last_fight": days_since,
        "decayed_win_rate": decayed_win_rate,

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

        "sig_str_defence": (
            max(0.0, min(1.0, 1.0 - safe_divide(
                sig_absorbed,
                sig_faced
            )))
            if sig_faced > 0
            else DEFAULT_STRIKE_DEFENCE
        ),

        "decayed_sig_str_defence": decayed_sig_defence,

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
            max(0.0, min(1.0, 1.0 - safe_divide(
                td_absorbed,
                td_faced
            )))
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


def get_fighter_features(history, current_date=None):

    features = {}

    features.update(
        get_career_features(history, current_date)
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
        "recent": deque(maxlen=5),
        "elo": DEFAULT_ELO,
        "last_fight_date": None,
    }
)


feature_names = [

    "total_fights",
    "wins",
    "losses",
    "win_rate",
    "elo",
    "is_debut",
    "days_since_last_fight",
    "decayed_win_rate",

    "sig_str_landed_per_fight",
    "sig_str_attempted_per_fight",
    "sig_str_accuracy",
    "sig_str_defence",
    "decayed_sig_str_defence",

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
            history_a,
            date
        )

        features_b = get_fighter_features(
            history_b,
            date
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

            "date": date,

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

            "sig_faced": row[
                "b_sigstr_attempted"
            ],

            "td_faced": row[
                "b_td_attempted"
            ],

            "td_absorbed": row[
                "b_td_landed"
            ],
        }


        b_fight_record = {

            "date": date,

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

            "sig_faced": row[
                "a_sigstr_attempted"
            ],

            "td_faced": row[
                "a_td_attempted"
            ],

            "td_absorbed": row[
                "a_td_landed"
            ],
        }


        # -------------------------------------------------
        # ROLLING ELO UPDATE (WITH METHOD FINISH MARGIN)
        # -------------------------------------------------

        r_a = histories[fighter_a]["elo"]
        r_b = histories[fighter_b]["elo"]
        expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        expected_b = 1.0 - expected_a
        score_a = 1.0 if a_won else 0.0
        score_b = 1.0 - score_a

        k_a = INITIAL_ELO_K if len(histories[fighter_a]["fights"]) < 3 else BASE_ELO_K
        k_b = INITIAL_ELO_K if len(histories[fighter_b]["fights"]) < 3 else BASE_ELO_K

        method_str = str(row.get("METHOD", "")).lower()
        if "ko" in method_str or "sub" in method_str:
            margin_mult = 1.2
        elif "split" in method_str or "majority" in method_str:
            margin_mult = 0.8
        else:
            margin_mult = 1.0

        histories[fighter_a]["elo"] = r_a + (k_a * margin_mult) * (score_a - expected_a)
        histories[fighter_b]["elo"] = r_b + (k_b * margin_mult) * (score_b - expected_b)

        histories[fighter_a]["last_fight_date"] = date
        histories[fighter_b]["last_fight_date"] = date


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

    profile = {
        "fighter": fighter_name,
        "last_fight_date": history["last_fight_date"],
    }

    profile.update(
        get_fighter_features(history, None)
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