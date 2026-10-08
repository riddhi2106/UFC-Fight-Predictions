# Predicting UFC Fight Outcomes with Machine Learning

Mini-project for **UE24CS352A – Machine Learning**
Problem Statement 74: *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes*

---

## Problem

Given two fighters about to meet, predict which one wins, using only information
available **before** the fight takes place.

This is a binary classification task. The input is the pair of fighters'
accumulated records and physical attributes; the output is which of the two wins.

The reference study for this problem statement is McQuaide, *Applying Machine
Learning Algorithms to Predict UFC Fight Outcomes* (CS229, Autumn 2019), which
reported roughly **60% accuracy** across four model families. Earlier published
work on the same task falls between 51.7% and 58.4%.

---

## Results

Evaluated on 1,761 fights from June 2023 to September 2026, none of which appear
in training.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | Log-Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.6246** | 0.6131 | 0.6463 | **0.6293** | **0.6693** | **0.2284** | **0.6482** |
| Gradient Boosting | 0.6076 | 0.5976 | 0.6244 | 0.6107 | 0.6488 | 0.2330 | 0.6582 |
| MLP | 0.5934 | 0.5950 | 0.5484 | 0.5707 | 0.6243 | 0.2494 | 0.7098 |
| Decision Tree | 0.5866 | 0.6012 | 0.4793 | 0.5333 | 0.6217 | 0.2433 | 0.6893 |

Majority-class baseline: **0.5071**. The best model beats it by **11.8 points** and
markedly outperforms the reference study across all metrics.

### Accuracy scales with confidence

The model is not equally sure about every fight, and it is right more often when
it is more sure:

| Model confidence | Accuracy | Fights |
|---|---:|---:|
| lowest | 0.5269 | 353 |
| low | 0.5739 | 352 |
| medium | 0.5881 | 352 |
| high | 0.7017 | 352 |
| highest | **0.7330** | 352 |

In the top confidence tier, it achieves **73.3% accuracy**, with the high+highest tiers
(top 40% of all test fights) averaging **71.7% accuracy**.

---

## Dataset

Raw data is scraped UFCStats records, stored under `data/raw/ufcstats/`:

| File | Contents |
|---|---|
| `ufc_event_details.csv` | Event names and dates |
| `ufc_fight_results.csv` | Bouts, outcomes, methods, weight classes |
| `ufc_fight_details.csv` | Fight metadata |
| `ufc_fight_stats.csv` | Round-by-round per-fighter statistics |

`data/raw/complete_ufc_data.csv` supplies static physical attributes (height,
reach, stance, date of birth). Its aggregated performance columns are **not**
used, because they are career totals that include fights occurring after any
given prediction date.

After aggregating rounds to fights and removing draws, no-contests, and fights
without dates, **8,755 fights** remain, spanning March 1994 to September 2026.

---

## Approach

### 1. Resolving the winner correctly

UFCStats states `OUTCOME` relative to the `BOUT` string: `W/L` means the fighter
*named first* in the bout won. But the two fighters are read out of the
round-level statistics table, whose row order matches the bout string only about
half the time.

Taking `OUTCOME` at face value therefore mislabels roughly half of all fights.
`src/build_fight_dataset.py` resolves the winner by matching fighter names
against both sides of the bout string, producing an `a_won` column that is
correct regardless of row order.

### 2. Chronological feature construction

Fights are processed in date order. For each fight, both fighters' features are
computed from their record **before** that fight. Histories are updated only
after every fight in an event has been turned into a feature row, so two fights
on the same card cannot leak into each other.

### 3. Matchup differences

Each fighter contributes 20 career and recent-form statistics. The model sees the
difference between them:

```
feature = fighter A's statistic − fighter B's statistic
```

Covering experience, wins and losses, win rate, striking volume and accuracy,
strikes absorbed, takedown volume, accuracy and defence, submission attempts,
reversals, control time, and three- and five-fight recent form.

### 4. Physical attributes

Height, reach, stance, and age differences are joined from the static attribute
table. These are fixed properties, so they introduce no leakage; age is computed
against each event date, giving the fighter's real age that night.

Coverage is incomplete (height 90%, stance 89%, reach 79%). Missing differences
are filled with `0.0` and paired with a `_known` flag, so the model can
distinguish "no advantage" from "unknown".

**35 features** total (incorporating rolling pre-fight Elo, exponential recency time decay, cage rust / layoff duration, UFC debutant flags, and significant strike defense).

### 5. Symmetrical Data Augmentation & Chronological Split

| Split | Period | Fights |
|---|---|---:|
| Train (original) | 1994-03-11 → 2023-05-20 | 6,994 |
| Train (augmented) | Symmetrically paired mirrors | 13,988 |
| Test (holdout) | 2023-06-03 → 2026-09-26 | 1,761 |

To eliminate orientation bias and enforce mathematical symmetry ($P(A \text{ beats } B) = 1 - P(B \text{ beats } A)$), training folds are augmented with mirrored pairings where antisymmetric differences and outcomes are negated. Hyperparameters are selected via `TimeSeriesSplit(n_splits=5)` scored on ROC-AUC, where each CV training fold is symmetrically augmented while validation folds contain only original, forward chronological bouts (zero leakage).

---

## What the models learned

Both model families independently place rolling Elo, exponential decay striking defense, and age near the top:

| Logistic Regression (scaled coefficients) | Gradient Boosting (importance) |
|---|---|
| `diff_elo` +0.461 | `diff_age` 0.250 |
| `diff_decayed_sig_str_defence` +0.322 | `diff_elo` 0.217 |
| `diff_age` −0.296 | `diff_decayed_sig_str_defence` 0.118 |
| `diff_sig_str_absorbed_per_fight` −0.277 | `diff_sig_str_absorbed_per_fight` 0.068 |
| `diff_decayed_win_rate` +0.273 | `diff_sig_str_landed_per_fight` 0.058 |

The physical coherence is remarkable: the fighter with superior Elo rating, higher recent striking defense, younger age, and lower absorbed damage wins consistently.

---

## Project structure

```text
UFC-Fight-Predictions/
├── data/
│   ├── raw/
│   │   ├── complete_ufc_data.csv        physical attributes
│   │   └── ufcstats/                    scraped source tables
│   └── processed/
│       ├── fight_level_data.csv         one row per fight
│       ├── ml_dataset.csv               model-ready features
│       └── fighter_profiles.csv         per-fighter career state
│
├── src/
│   ├── inspect_ufcstats.py              source inspection
│   ├── validate_ufcstats.py             source validation
│   ├── check_event_dates.py             date checks
│   ├── data_inspection.py               dataset profiling
│   ├── check_history.py                 history sanity checks
│   ├── build_fight_dataset.py           rounds → fights, winner resolution
│   ├── feature_engineering.py           pre-fight features, profiles
│   ├── train.py                         tuning and training
│   ├── evaluate.py                      metrics and confusion matrices
│   ├── analysis.py                      importance, calibration, errors
│   └── predict.py                       single-matchup demo
│
├── results/
│   ├── models/                          four trained models
│   ├── figures/                         all plots
│   ├── analysis/                        importance and error tables
│   ├── model_comparison.csv
│   ├── model_predictions.csv
│   └── tuning_results.csv
│
├── README.md
└── requirements.txt
```

---

## Setup

Requires Python 3.9 or newer.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

Run from the repository root, in order. Steps 1–3 take a few seconds each;
step 3 includes the grid search and takes about 35 seconds.

```bash
# 1. Aggregate round-level stats into fights and resolve winners
python3 src/build_fight_dataset.py

# 2. Build pre-fight features and fighter profiles
python3 src/feature_engineering.py

# 3. Tune and train all four models
python3 src/train.py

# 4. Score the models and write confusion matrices
python3 src/evaluate.py

# 5. Feature importance, calibration, and error analysis
python3 src/analysis.py
```

## Predicting a single fight

```bash
python3 src/predict.py "Islam Makhachev" "Charles Oliveira"
```

```text
==========================================================
Islam Makhachev  vs  Charles Oliveira
==========================================================
Model: logistic_regression
Fight date: 2026-10-05

Islam Makhachev                   64.4%
Charles Oliveira                  35.6%

Predicted winner: Islam Makhachev  (64.4%)
```

Misspelled names return suggestions. Options:

```bash
python3 src/predict.py "Jon Jones" "Tom Aspinall" --model gradient_boosting
python3 src/predict.py "Jon Jones" "Tom Aspinall" --date 2026-12-01
python3 src/predict.py --self-check      # verify feature construction
```

`--self-check` confirms that swapping the two fighters negates every difference
feature while leaving the pairing flags unchanged, so a prediction never depends
on which name is typed first.

Profiles are end-of-dataset career states, so `predict.py` is valid for
hypothetical or upcoming fights, not for re-scoring fights already in the data.

---

## Limitations

- Accuracy near 61% reflects a genuinely high-variance sport, not an
  under-trained model. A single punch can end a fight regardless of record.
- Reach is missing for 21% of fighters, so that feature is partly inactive.
- No injury, camp, late-replacement, or layoff information is available.
- Fighter identity is matched by name string; a renamed fighter would appear twice.
- Betting odds are present in the raw data but deliberately unused, since they
  encode the market's answer rather than fighter performance.
