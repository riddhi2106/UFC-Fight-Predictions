import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/processed/ml_dataset.csv"
MODEL_DIR = "results/models"
PREDICTION_PATH = "results/model_predictions.csv"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["DATE"] = pd.to_datetime(df["DATE"])

# Sort explicitly by date
df = df.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)


# --------------------------------------------------
# Select matchup-difference features
# --------------------------------------------------

feature_columns = [
    col for col in df.columns
    if col.startswith("diff_")
]

X = df[feature_columns]
y = df["target"]


# --------------------------------------------------
# Chronological 80/20 split
# --------------------------------------------------

cutoff_date = pd.Timestamp("2023-06-03")

train_mask = df["DATE"] < cutoff_date
test_mask = df["DATE"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

train_dates = df.loc[train_mask, "DATE"]
test_dates = df.loc[test_mask, "DATE"]


print("=" * 60)
print("UFC FIGHT PREDICTION - MODEL TRAINING")
print("=" * 60)

print(f"\nTotal fights: {len(df)}")
print(f"Training fights: {len(X_train)}")
print(f"Testing fights: {len(X_test)}")

print(
    f"\nTraining period: "
    f"{train_dates.min().date()} → {train_dates.max().date()}"
)

print(
    f"Testing period: "
    f"{test_dates.min().date()} → {test_dates.max().date()}"
)

print(f"\nNumber of features: {len(feature_columns)}")


# --------------------------------------------------
# Define models
# --------------------------------------------------

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "decision_tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "gradient_boosting": GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3
    ),

    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42
            )
        )
    ])
}


# --------------------------------------------------
# Train models and generate predictions
# --------------------------------------------------

prediction_data = df.loc[test_mask, [
    "DATE",
    "EVENT",
    "BOUT",
    "fighter_a",
    "fighter_b",
    "target"
]].copy()

prediction_data = prediction_data.reset_index(drop=True)

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")

    prediction_data[f"{name}_prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        prediction_data[f"{name}_probability"] = probabilities

    model_path = os.path.join(
        MODEL_DIR,
        f"{name}.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

prediction_data.to_csv(
    PREDICTION_PATH,
    index=False
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nSaved predictions to: {PREDICTION_PATH}")
print(f"Saved models to: {MODEL_DIR}/")