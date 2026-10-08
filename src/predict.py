"""
Predict the outcome of a single UFC matchup.

This is the live-demo entry point. It scores a fight that
is not in the dataset, using each fighter's end-of-dataset
career profile plus their static physical attributes.

    python3 src/predict.py "Islam Makhachev" "Charles Oliveira"

Because the profiles are end-of-dataset values, this is
only valid for predicting a FUTURE fight. Re-scoring a
past fight this way would use information recorded after
that fight took place.
"""

import argparse
import difflib
import sys

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROFILE_PATH = Path("data/processed/fighter_profiles.csv")
DATASET_PATH = Path("data/processed/ml_dataset.csv")
MODEL_DIR = Path("results/models")

DEFAULT_MODEL = "logistic_regression"

# Physical features are handled separately from the career
# history features because they need a _known flag and, in
# the case of age, the fight date.
PHYSICAL_NAMES = ["age", "height", "reach", "southpaw"]


def load_feature_order():
    """Read the exact model feature order from the dataset."""

    header = pd.read_csv(DATASET_PATH, nrows=0)

    return [
        column
        for column in header.columns
        if column.startswith("diff_")
    ]


def resolve_fighter(name, profiles):
    """Match a typed name to a fighter, or explain why not."""

    name = name.strip()

    if name in profiles.index:
        return name

    lowered = {
        str(index).lower(): index
        for index in profiles.index
    }

    if name.lower() in lowered:
        return lowered[name.lower()]

    suggestions = difflib.get_close_matches(
        name,
        [str(index) for index in profiles.index],
        n=5,
        cutoff=0.6,
    )

    message = f"Fighter not found: {name!r}"

    if suggestions:
        message += "\n  Did you mean: " + ", ".join(suggestions)

    raise SystemExit(message)


def physical_values(profile, fight_date):
    """Return this fighter's physical attributes as floats."""

    date_of_birth = pd.to_datetime(
        profile.get("dob"),
        errors="coerce",
    )

    if pd.isna(date_of_birth):
        age = np.nan
    else:
        age = (fight_date - date_of_birth).days / 365.25

        # Mirrors the guard in feature_engineering.py.
        if not (15 < age < 60):
            age = np.nan

    return {
        "age": age,
        "height": profile.get("height", np.nan),
        "reach": profile.get("reach", np.nan),
        "southpaw": profile.get("is_southpaw", np.nan),
    }


def build_features(profile_a, profile_b, fight_date, feature_order):
    """Build the model's feature vector for one matchup."""

    values = {}

    # Career history differences.
    for column in feature_order:

        if not column.startswith("diff_"):
            continue

        name = column[len("diff_"):]

        if name in profile_a.index and name in profile_b.index:
            values[column] = (
                float(profile_a[name]) - float(profile_b[name])
            )

    # Dynamic layoff calculation based on fight date
    def calculate_layoff(profile):
        lfd = profile.get("last_fight_date")
        if pd.isna(lfd) or lfd is None:
            return 365.0
        dt = pd.to_datetime(lfd, errors="coerce")
        if pd.isna(dt):
            return 365.0
        return min(max(0.0, float((fight_date - dt).days)), 1000.0)

    if "diff_days_since_last_fight" in feature_order:
        values["diff_days_since_last_fight"] = (
            calculate_layoff(profile_a) - calculate_layoff(profile_b)
        )

    # Physical differences, with their known flags.
    physical_a = physical_values(profile_a, fight_date)
    physical_b = physical_values(profile_b, fight_date)

    for name in PHYSICAL_NAMES:

        difference = physical_a[name] - physical_b[name]
        known = not pd.isna(difference)

        values[f"diff_{name}"] = float(difference) if known else 0.0
        values[f"diff_{name}_known"] = int(known)

    values["diff_stance_mismatch"] = int(
        not pd.isna(physical_a["southpaw"])
        and not pd.isna(physical_b["southpaw"])
        and physical_a["southpaw"] != physical_b["southpaw"]
    )

    missing = [
        column
        for column in feature_order
        if column not in values
    ]

    if missing:
        raise SystemExit(
            "Could not build these features: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [[values[column] for column in feature_order]],
        columns=feature_order,
    )


def self_check(profiles, feature_order):
    """
    Feature construction must be order-aware.

    Swapping the two fighters has to negate every genuine
    difference, while the _known and stance_mismatch flags
    describe the pairing itself and must stay put. A silent
    break here would make predictions depend on which name
    was typed first.
    """

    names = list(profiles.index[:2])
    date = pd.Timestamp("2026-01-01")

    forward = build_features(
        profiles.loc[names[0]],
        profiles.loc[names[1]],
        date,
        feature_order,
    ).iloc[0]

    reverse = build_features(
        profiles.loc[names[1]],
        profiles.loc[names[0]],
        date,
        feature_order,
    ).iloc[0]

    for column in feature_order:

        symmetric = (
            column.endswith("_known")
            or column == "diff_stance_mismatch"
        )

        if symmetric:
            assert forward[column] == reverse[column], column
        else:
            assert np.isclose(
                forward[column], -reverse[column]
            ), column

    print(f"self-check passed on {len(feature_order)} features")


def main():

    parser = argparse.ArgumentParser(
        description="Predict a UFC fight outcome."
    )

    parser.add_argument("fighter_a", nargs="?", help="First fighter")
    parser.add_argument("fighter_b", nargs="?", help="Second fighter")

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Fight date YYYY-MM-DD, used for age (default: today)",
    )

    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Verify feature construction and exit",
    )

    arguments = parser.parse_args()

    profiles = pd.read_csv(PROFILE_PATH).set_index("fighter")
    feature_order = load_feature_order()

    if arguments.self_check:
        self_check(profiles, feature_order)
        return

    if not arguments.fighter_a or not arguments.fighter_b:
        parser.error("two fighter names are required")

    fight_date = (
        pd.Timestamp(arguments.date)
        if arguments.date
        else pd.Timestamp.today().normalize()
    )

    name_a = resolve_fighter(arguments.fighter_a, profiles)
    name_b = resolve_fighter(arguments.fighter_b, profiles)

    if name_a == name_b:
        raise SystemExit("A fighter cannot fight themselves.")

    model_path = MODEL_DIR / f"{arguments.model}.joblib"

    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}. "
            "Run src/train.py first."
        )

    model = joblib.load(model_path)

    features = build_features(
        profiles.loc[name_a],
        profiles.loc[name_b],
        fight_date,
        feature_order,
    )

    probability_a = float(model.predict_proba(features)[0, 1])
    probability_b = 1.0 - probability_a

    winner = name_a if probability_a >= 0.5 else name_b
    confidence = max(probability_a, probability_b)

    print()
    print("=" * 58)
    print(f"{name_a}  vs  {name_b}")
    print("=" * 58)
    print(f"Model: {arguments.model}")
    print(f"Fight date: {fight_date.date()}")
    print()
    print(f"{name_a:<32} {probability_a:>6.1%}")
    print(f"{name_b:<32} {probability_b:>6.1%}")
    print()
    print(f"Predicted winner: {winner}  ({confidence:.1%})")

    if confidence < 0.55:
        print(
            "\nNote: below 55% is the model's coin-flip range. "
            "Accuracy there is about 53%."
        )

    print()
    print("Key differences (fighter A minus fighter B):")

    for name in [
        "elo",
        "decayed_win_rate",
        "win_rate",
        "age",
        "days_since_last_fight",
        "reach",
        "sig_str_defence",
        "td_defence",
        "sig_str_absorbed_per_fight",
    ]:

        column = f"diff_{name}"

        if column in features.columns:
            print(f"  {name:<30} {features.iloc[0][column]:>8.2f}")


if __name__ == "__main__":
    main()
