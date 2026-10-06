"""
Model interpretation and error analysis.

Runs after train.py and evaluate.py. Reads the saved
models and the saved test-set predictions, so it never
refits anything and never touches the training data.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)


DATA_PATH = Path("data/processed/ml_dataset.csv")
PREDICTION_PATH = Path("results/model_predictions.csv")
MODEL_DIR = Path("results/models")

FIGURE_DIR = Path("results/figures")
ANALYSIS_DIR = Path("results/analysis")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "gradient_boosting",
    "mlp",
]

BEST_MODEL = "logistic_regression"

report_lines = []


def record(line=""):
    """Print a line and keep it for the written report."""

    print(line)
    report_lines.append(str(line))


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_PATH)
df["DATE"] = pd.to_datetime(df["DATE"])

predictions = pd.read_csv(PREDICTION_PATH)
predictions["DATE"] = pd.to_datetime(predictions["DATE"])

feature_columns = [
    column
    for column in df.columns
    if column.startswith("diff_")
]

# WEIGHTCLASS is needed for the per-division breakdown and
# is not carried in the predictions file.
predictions = predictions.merge(
    df[["DATE", "EVENT", "BOUT", "WEIGHTCLASS"]],
    on=["DATE", "EVENT", "BOUT"],
    how="left",
)

models = {
    name: joblib.load(MODEL_DIR / f"{name}.joblib")
    for name in MODEL_NAMES
}

record("=" * 60)
record("UFC FIGHT PREDICTION - ANALYSIS")
record("=" * 60)
record()
record(f"Test fights: {len(predictions)}")
record(f"Features: {len(feature_columns)}")

baseline = max(
    predictions["target"].mean(),
    1 - predictions["target"].mean(),
)

record(f"Majority-class baseline: {baseline:.4f}")
record()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
#
# Logistic regression coefficients are read off the scaled
# pipeline, so every feature is on the same scale and the
# magnitudes are directly comparable.

logistic_model = models["logistic_regression"].named_steps["model"]

coefficients = pd.Series(
    logistic_model.coef_[0],
    index=feature_columns,
).sort_values(key=abs, ascending=False)

boosting_importance = pd.Series(
    models["gradient_boosting"].feature_importances_,
    index=feature_columns,
).sort_values(ascending=False)

record("-" * 60)
record("TOP 10 LOGISTIC REGRESSION COEFFICIENTS (scaled)")
record("-" * 60)
record(coefficients.head(10).round(4).to_string())
record()

record("-" * 60)
record("TOP 10 GRADIENT BOOSTING IMPORTANCES")
record("-" * 60)
record(boosting_importance.head(10).round(4).to_string())
record()

coefficients.to_csv(ANALYSIS_DIR / "logistic_coefficients.csv")
boosting_importance.to_csv(ANALYSIS_DIR / "gradient_boosting_importance.csv")

figure, axes = plt.subplots(1, 2, figsize=(14, 6))

top_coefficients = coefficients.head(12).iloc[::-1]

axes[0].barh(
    top_coefficients.index,
    top_coefficients.values,
    color=np.where(top_coefficients.values > 0, "#2a9d8f", "#e76f51"),
)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("Logistic regression coefficients (scaled)")
axes[0].set_xlabel("Coefficient")

top_importance = boosting_importance.head(12).iloc[::-1]

axes[1].barh(
    top_importance.index,
    top_importance.values,
    color="#264653",
)
axes[1].set_title("Gradient boosting feature importance")
axes[1].set_xlabel("Importance")

figure.tight_layout()
figure.savefig(FIGURE_DIR / "feature_importance.png", dpi=150)
plt.close(figure)


# =========================================================
# ROC CURVES
# =========================================================

figure, axis = plt.subplots(figsize=(7, 6))

for name in MODEL_NAMES:

    probability_column = f"{name}_probability"

    if probability_column not in predictions.columns:
        continue

    false_positive, true_positive, _ = roc_curve(
        predictions["target"],
        predictions[probability_column],
    )

    area = roc_auc_score(
        predictions["target"],
        predictions[probability_column],
    )

    axis.plot(
        false_positive,
        true_positive,
        label=f"{name} (AUC {area:.3f})",
    )

axis.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
axis.set_xlabel("False positive rate")
axis.set_ylabel("True positive rate")
axis.set_title("ROC curves on the chronological test set")
axis.legend(loc="lower right")

figure.tight_layout()
figure.savefig(FIGURE_DIR / "roc_curves.png", dpi=150)
plt.close(figure)


# =========================================================
# CALIBRATION
# =========================================================
#
# A model can rank fights well and still state badly
# scaled probabilities. Calibration shows whether a stated
# 70% actually wins about 70% of the time.

probabilities = predictions[f"{BEST_MODEL}_probability"]

observed, predicted = calibration_curve(
    predictions["target"],
    probabilities,
    n_bins=10,
    strategy="quantile",
)

figure, axis = plt.subplots(figsize=(6, 6))

axis.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
axis.plot(predicted, observed, "o-", color="#2a9d8f", label=BEST_MODEL)

axis.set_xlabel("Predicted probability")
axis.set_ylabel("Observed win rate")
axis.set_title(f"Calibration - {BEST_MODEL}")
axis.legend()

figure.tight_layout()
figure.savefig(FIGURE_DIR / "calibration.png", dpi=150)
plt.close(figure)


# =========================================================
# ERROR ANALYSIS: CONFIDENCE
# =========================================================
#
# Confidence is the distance of the stated probability
# from 0.5. If the model is behaving sensibly, the fights
# it is most confident about should be the fights it gets
# right most often.

confidence = (probabilities - 0.5).abs()
correct = (
    predictions[f"{BEST_MODEL}_prediction"] == predictions["target"]
)

confidence_frame = pd.DataFrame({
    "confidence": confidence,
    "correct": correct,
})

confidence_frame["bucket"] = pd.qcut(
    confidence_frame["confidence"],
    5,
    labels=["lowest", "low", "medium", "high", "highest"],
)

confidence_accuracy = (
    confidence_frame
    .groupby("bucket", observed=True)["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "fights"})
)

record("-" * 60)
record("ACCURACY BY CONFIDENCE BUCKET")
record("-" * 60)
record(confidence_accuracy.round(4).to_string())
record()

figure, axis = plt.subplots(figsize=(7, 5))

axis.bar(
    confidence_accuracy.index.astype(str),
    confidence_accuracy["accuracy"],
    color="#264653",
)
axis.axhline(
    baseline,
    color="#e76f51",
    linestyle="--",
    label=f"Majority baseline ({baseline:.3f})",
)

axis.set_ylim(0.4, 0.8)
axis.set_ylabel("Accuracy")
axis.set_xlabel("Model confidence")
axis.set_title("Accuracy rises with model confidence")
axis.legend()

figure.tight_layout()
figure.savefig(FIGURE_DIR / "accuracy_by_confidence.png", dpi=150)
plt.close(figure)


# =========================================================
# ERROR ANALYSIS: WEIGHT CLASS AND TIME
# =========================================================

weight_accuracy = (
    predictions
    .assign(correct=correct)
    .groupby("WEIGHTCLASS")["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "fights"})
    .query("fights >= 30")
    .sort_values("accuracy", ascending=False)
)

record("-" * 60)
record("ACCURACY BY WEIGHT CLASS (30+ fights)")
record("-" * 60)
record(weight_accuracy.round(4).to_string())
record()

weight_accuracy.to_csv(ANALYSIS_DIR / "accuracy_by_weightclass.csv")

yearly_accuracy = (
    predictions
    .assign(correct=correct, year=predictions["DATE"].dt.year)
    .groupby("year")["correct"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "accuracy", "count": "fights"})
)

record("-" * 60)
record("ACCURACY BY YEAR")
record("-" * 60)
record(yearly_accuracy.round(4).to_string())
record()


# =========================================================
# EDA: HOW THE STRONGEST FEATURES SEPARATE OUTCOMES
# =========================================================

eda_features = ["diff_age", "diff_win_rate", "diff_reach"]

figure, axes = plt.subplots(1, len(eda_features), figsize=(15, 4.5))

for axis, feature in zip(axes, eda_features):

    # Physical features carry a _known flag. Rows where the
    # attribute was missing were filled with 0.0, which would
    # otherwise show up as a false spike at the centre.
    known_column = f"{feature}_known"

    subset = (
        df[df[known_column] == 1]
        if known_column in df.columns
        else df
    )

    axis.hist(
        subset.loc[subset["target"] == 1, feature],
        bins=40,
        alpha=0.6,
        label="Fighter A won",
        color="#2a9d8f",
        density=True,
    )
    axis.hist(
        subset.loc[subset["target"] == 0, feature],
        bins=40,
        alpha=0.6,
        label="Fighter A lost",
        color="#e76f51",
        density=True,
    )

    axis.set_title(feature)
    axis.legend(fontsize=8)

figure.tight_layout()
figure.savefig(FIGURE_DIR / "feature_distributions.png", dpi=150)
plt.close(figure)


# =========================================================
# SAVE REPORT
# =========================================================

report_path = ANALYSIS_DIR / "analysis_report.txt"

report_path.write_text("\n".join(report_lines))

record("=" * 60)
record(f"Saved report to: {report_path}")
record("Saved figures to: results/figures/")
