# UFC Fight Prediction Using Machine Learning

## Project Overview

This project investigates whether historical UFC fighter performance can be used to predict the outcome of future UFC fights using machine learning.

The project focuses on pre-fight information only. Fighter statistics are constructed chronologically so that information from a fight is never used to predict that same fight.

---

## Dataset

The project uses UFCStats fight-level data containing:

- Fight results
- Fighter identities
- Fight-level statistics
- Round-level statistics
- Event information and dates

The raw data is stored under:

`data/raw/ufcstats/`

The main source files are:

- `ufc_event_details.csv`
- `ufc_fight_details.csv`
- `ufc_fight_results.csv`
- `ufc_fight_stats.csv`

The original combined UFC dataset is retained as a reference under:

`data/raw/complete_ufc_data.csv`

It is not used directly for model training because its fighter performance attributes represent aggregated information that can contain information from fights occurring after the prediction date.

---

## Data Processing

The raw round-level UFCStats data is aggregated into a fight-level dataset.

The resulting dataset:

- Contains 8,755 completed fights
- Removes fights without usable fighter-level statistics
- Removes draws and no-contests
- Contains exactly two fighters per fight
- Is ordered chronologically

The processed fight-level dataset is:

`data/processed/fight_level_data.csv`

---

## Temporal Feature Engineering

For every fight, fighter statistics are calculated using only information available before that event.

The feature groups include:

- Career experience
- Wins and losses
- Win rate
- Significant striking performance
- Striking accuracy
- Takedown performance
- Takedown accuracy
- Submission attempts
- Reversals
- Control time
- Significant strikes absorbed
- Takedowns faced
- Takedown defence
- Recent three-fight form
- Recent five-fight form

For each fight, the primary modelling representation is the difference between Fighter A and Fighter B:

`Fighter A historical statistic - Fighter B historical statistic`

This produces 20 matchup-difference features.

### Temporal Leakage Prevention

Historical fighter information is calculated before the current event.

All fights within the same event use the fighters' histories from before that event. Fighter histories are updated only after all fights from the event have been converted into feature rows.

This prevents one fight from an event from influencing the features of another fight from the same event.

The final machine-learning dataset is:

`data/processed/ml_dataset.csv`

It contains:

- 8,755 fights
- 67 columns
- 20 primary matchup-difference features
- No missing feature values

---

## Train/Test Split

A chronological split is used rather than a random split.

### Training Set

- Period: 1994-03-11 → 2023-05-20
- Fights: 6,994

### Test Set

- Period: 2023-06-03 → 2026-09-26
- Fights: 1,761

The test set therefore represents a later period than the training set.

This setup is intended to better reflect the task of predicting future UFC fight outcomes.

---

## Machine Learning Models

Four baseline classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. Gradient Boosting
4. Multi-Layer Perceptron (MLP)

The models are implemented in:

`src/train.py`

Training is performed using the chronological training set only.

For models requiring feature scaling, the scaler is fitted only on the training data.

---

## Baseline Evaluation

The baseline evaluation is implemented in:

`src/evaluate.py`

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

The initial test-set accuracy results are:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 56.27% |
| Decision Tree | 55.65% |
| Gradient Boosting | 56.22% |
| MLP | 51.79% |

These are baseline results from the initial model configurations and are not treated as final optimised performance.

---

## Project Structure

```text
UFC-Fight-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── complete_ufc_data.csv
│   │   └── ufcstats/
│   │       ├── ufc_event_details.csv
│   │       ├── ufc_fight_details.csv
│   │       ├── ufc_fight_results.csv
│   │       └── ufc_fight_stats.csv
│   │
│   └── processed/
│       ├── fight_level_data.csv
│       └── ml_dataset.csv
│
├── src/
│   ├── data_inspection.py
│   ├── check_history.py
│   ├── inspect_ufcstats.py
│   ├── validate_ufcstats.py
│   ├── check_event_dates.py
│   ├── build_fight_dataset.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── models/
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── gradient_boosting.joblib
│   │   └── mlp.joblib
│   │
│   ├── figures/
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── gradient_boosting_confusion_matrix.png
│   │   └── mlp_confusion_matrix.png
│   │
│   ├── model_predictions.csv
│   └── model_comparison.csv
│
├── README.md
└── requirements.txt
```

## Running the Project

### 1. Install dependencies

Activate the virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Build the fight-level dataset

```bash
python3 src/build_fight_dataset.py
```

### 3. Generate historical pre-fight features

```bash
python3 src/feature_engineering.py
```

### 4. Train the baseline models

```bash
python3 src/train.py
```

### 5. Evaluate the models

```bash
python3 src/evaluate.py
```

---

## Current Status

The data-processing and baseline machine-learning pipeline has been completed.

The current pipeline includes:

- UFCStats data validation
- Fight-level dataset construction
- Chronological fighter-history construction
- Temporal leakage prevention
- Pre-fight feature engineering
- Matchup-difference features
- Chronological train/test splitting
- Four baseline machine-learning models
- Baseline evaluation and confusion matrices

The next stage of the project is **model improvement and deeper analysis**, including:

- Hyperparameter tuning
- Feature interpretation
- Error analysis
- Model comparison
- Prediction analysis