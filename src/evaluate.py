import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PREDICTION_PATH = "results/model_predictions.csv"
FIGURE_DIR = "results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

df = pd.read_csv(PREDICTION_PATH)

y_true = df["target"]


# --------------------------------------------------
# Model definitions
# --------------------------------------------------

models = {
    "Logistic Regression": {
        "prediction": "logistic_regression_prediction",
        "probability": "logistic_regression_probability"
    },
    "Decision Tree": {
        "prediction": "decision_tree_prediction",
        "probability": "decision_tree_probability"
    },
    "Gradient Boosting": {
        "prediction": "gradient_boosting_prediction",
        "probability": "gradient_boosting_probability"
    },
    "MLP": {
        "prediction": "mlp_prediction",
        "probability": "mlp_probability"
    }
}


# --------------------------------------------------
# Majority-class baseline
# --------------------------------------------------

majority_class = y_true.mode()[0]
majority_predictions = [majority_class] * len(y_true)

majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)

print("=" * 70)
print("UFC FIGHT PREDICTION - MODEL EVALUATION")
print("=" * 70)

print("\nTEST SET")
print(f"Number of fights: {len(y_true)}")
print(f"Class 1 (A wins): {(y_true == 1).sum()}")
print(f"Class 0 (B wins): {(y_true == 0).sum()}")

print(
    f"\nMajority-class baseline accuracy: "
    f"{majority_accuracy:.4f}"
)


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

results = []

for name, columns in models.items():

    predictions = df[columns["prediction"]]

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    probability_column = columns["probability"]

    if probability_column is not None:
        probabilities = df[probability_column]

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    else:
        roc_auc = None

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# Comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_path = "results/model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print(f"\nSaved comparison table to: {results_path}")


# --------------------------------------------------
# Confusion matrices
# --------------------------------------------------

for name, columns in models.items():

    predictions = df[columns["prediction"]]

    cm = confusion_matrix(
        y_true,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["B wins", "A wins"]
    )

    display.plot()

    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    path = os.path.join(
        FIGURE_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


print("\nEvaluation complete.")