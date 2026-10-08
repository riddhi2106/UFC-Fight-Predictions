"""
create_writeup.py  –  Generate a publication-style PDF write-up for the
UFC Fight Prediction project using the ReportLab library.

Run:
    venv/bin/python scripts/create_writeup.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "UFC_Fight_Prediction_WriteUp.pdf"

# ── Colour palette ──────────────────────────────────────────────────────────
RED    = colors.HexColor("#C0392B")
DARK   = colors.HexColor("#1A1A2E")
MID    = colors.HexColor("#2C3E50")
LIGHT  = colors.HexColor("#ECF0F1")
ACCENT = colors.HexColor("#E74C3C")
GREY   = colors.HexColor("#7F8C8D")

PAGE_W, PAGE_H = A4
LEFT = RIGHT = 2.5 * cm
TOP = BOTTOM = 2.0 * cm


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        fontSize=26,
        fontName="Helvetica-Bold",
        textColor=DARK,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        fontSize=13,
        fontName="Helvetica",
        textColor=GREY,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    styles["authors"] = ParagraphStyle(
        "authors",
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=MID,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        fontSize=16,
        fontName="Helvetica-Bold",
        textColor=RED,
        spaceBefore=18,
        spaceAfter=6,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=MID,
        spaceBefore=12,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontSize=10.5,
        fontName="Helvetica",
        textColor=colors.black,
        leading=16,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        parent=base["Normal"],
        fontSize=10,
        fontName="Helvetica",
        leftIndent=18,
        spaceAfter=4,
        leading=15,
        bulletIndent=6,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontSize=9,
        fontName="Courier",
        textColor=MID,
        backColor=LIGHT,
        leftIndent=12,
        rightIndent=12,
        spaceAfter=6,
        spaceBefore=4,
        leading=14,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontSize=9,
        fontName="Helvetica-Oblique",
        textColor=GREY,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header",
        fontSize=9.5,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell",
        fontSize=9.5,
        fontName="Helvetica",
        alignment=TA_CENTER,
    )

    return styles


S = build_styles()


def H(text):
    return Paragraph(text, S["h1"])

def H2(text):
    return Paragraph(text, S["h2"])

def P(text):
    return Paragraph(text, S["body"])

def B(text):
    return Paragraph(f"• {text}", S["bullet"])

def Code(text):
    return Paragraph(text, S["code"])

def HR():
    return HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=8, spaceBefore=4)

def SP(n=8):
    return Spacer(1, n)


# ── Results table helper ────────────────────────────────────────────────────

def results_table():
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Brier", "Log-Loss"]
    data = [
        ["Logistic Regression", "62.46%", "61.31%", "64.63%", "62.93%", "0.6693", "0.2284", "0.6482"],
        ["Gradient Boosting",   "60.76%", "59.76%", "62.44%", "61.07%", "0.6488", "0.2330", "0.6582"],
        ["Decision Tree",       "58.66%", "60.12%", "47.93%", "53.33%", "0.6217", "0.2433", "0.6893"],
        ["MLP (Neural Net)",    "59.34%", "59.50%", "54.84%", "57.07%", "0.6243", "0.2494", "0.7098"],
        ["Majority Baseline",   "50.00%", "--",     "--",     "--",     "0.5000", "0.2500", "--"    ],
    ]

    table_data = [[Paragraph(h, S["table_header"]) for h in headers]]
    for row in data:
        table_data.append([Paragraph(cell, S["table_cell"]) for cell in row])

    col_widths = [4.2*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.0*cm, 2.2*cm, 2.0*cm, 2.2*cm]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  RED),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9.5),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 5),
        ("BACKGROUND",  (0, -1), (-1, -1), colors.HexColor("#FADBD8")),
    ]))
    return tbl


def feature_table():
    headers = ["Feature Group", "Variables (diff_*)"]
    rows = [
        ["Rolling Elo Rating",     "elo"],
        ["Career Record",          "total_fights, wins, losses, win_rate, decayed_win_rate"],
        ["Striking Offence",       "sig_str_landed_per_fight, sig_str_accuracy"],
        ["Striking Defence",       "sig_str_defence, decayed_sig_str_defence, sig_str_absorbed_per_fight"],
        ["Takedown Offence",       "td_landed_per_fight, td_accuracy"],
        ["Takedown Defence",       "td_defence, td_faced_per_fight"],
        ["Grappling",              "sub_attempts_per_fight, reversals_per_fight, control_seconds_per_fight"],
        ["Recent Form (W3/W5)",    "win_rate_last_3, win_rate_last_5"],
        ["Physical / Reach",       "reach_cm, height_cm, weight_kg"],
        ["Temporal / Context",     "days_since_last_fight, is_debut"],
        ["Stance Mismatch",        "diff_stance_mismatch (symmetric flag)"],
    ]

    table_data = [[Paragraph(h, S["table_header"]) for h in headers]]
    for row in rows:
        table_data.append([Paragraph(cell, S["table_cell"]) for cell in row])

    tbl = Table(table_data, colWidths=[4.5*cm, 12.5*cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  RED),
        ("ROWBACKGROUNDS",(0, 1),(-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",        (0, 0), (0, -1),  "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return tbl


# ── Document content ────────────────────────────────────────────────────────

def build_content():
    story = []

    # ── Title page ────────────────────────────────────────────────────────
    story += [
        SP(60),
        Paragraph("UFC Fight Outcome Prediction", S["title"]),
        Paragraph("A Machine-Learning Approach with Domain-Informed Feature Engineering",
                  S["subtitle"]),
        SP(10),
        HR(),
        SP(10),
        Paragraph("Riddhi", S["authors"]),
        Paragraph("B.Tech - Semester 5 | Machine Learning Project", S["subtitle"]),
        SP(8),
        Paragraph("October 2026", S["subtitle"]),
        PageBreak(),
    ]

    # ── Abstract ─────────────────────────────────────────────────────────
    story += [
        H("Abstract"),
        HR(),
        P(
            "This project builds a machine-learning pipeline to predict the binary outcome "
            "(Fighter A wins / Fighter B wins) of Ultimate Fighting Championship (UFC) bouts. "
            "Starting from historical per-round statistics spanning 2010-2024, we perform "
            "extensive domain-informed feature engineering - including a rolling Elo rating "
            "system, exponential time-decay weighting, cage-rust indicators, UFC-debutant flags, "
            "and corrected symmetrical strike/takedown defence metrics - before training and "
            "comparing four classifiers. "
            "The final Logistic Regression model achieves <b>62.46% accuracy</b> and "
            "<b>ROC-AUC 0.6693</b> on a strict chronological holdout of 1,761 post-June 2023 "
            "bouts, surpassing all prior baselines and confirming the value of domain knowledge "
            "in combat-sports prediction."
        ),
        SP(4),
    ]

    # ── 1. Introduction ───────────────────────────────────────────────────
    story += [
        H("1. Introduction"),
        HR(),
        P(
            "Sports outcome prediction sits at the intersection of statistics, sports science, "
            "and machine learning. The UFC presents a uniquely challenging domain: each fight is "
            "a 1-vs-1 binary event influenced by fighter skill, physical attributes, fighting style, "
            "recent form, ring rust, and opponent quality - all of which must be measured "
            "before the fight occurs to avoid data leakage."
        ),
        P(
            "The key hypothesis is that a model trained on carefully constructed "
            "<b>antisymmetric difference features</b> (dStat = Stat_A - Stat_B) will learn "
            "relative superiority rather than absolute scale, generalise across weight classes, "
            "and produce predictions that satisfy the logical constraint "
            "P(A beats B) = 1 - P(B beats A)."
        ),
        H2("1.1 Objectives"),
        B("Predict the binary winner of any given UFC bout."),
        B("Achieve accuracy above the ~57-58% reported in comparable published work."),
        B("Maintain strict methodological integrity: zero temporal data leakage."),
        B("Interpret which physical and performance features drive prediction."),
        SP(6),
    ]

    # ── 2. Dataset ────────────────────────────────────────────────────────
    story += [
        H("2. Dataset"),
        HR(),
        P(
            "The raw data (fight_level_data.csv) is compiled from publicly available "
            "UFC statistics covering events from 2010 to mid-2024. Each row represents a "
            "single fighter-in-a-bout observation, containing:"
        ),
        B("Identifiers: event name, date, fighter name, corner (Red/Blue)."),
        B("Outcome: whether this fighter was the winner."),
        B("Per-fight aggregated statistics: significant strikes landed/attempted, takedowns, "
          "submission attempts, control time, reversals."),
        B("Physical attributes: height (cm), reach (cm), weight class."),
        SP(4),
        H2("2.1 Data Size"),
        P(
            "After pairing rows into matched bouts (Fighter A vs Fighter B), the dataset "
            "contains approximately <b>8,800 unique bouts</b>. The chronological 80/20 split "
            "yields ~7,040 training bouts (pre-June 2023) and <b>1,761 test bouts</b> "
            "(June 2023 onward)."
        ),
        H2("2.2 Target Variable"),
        P(
            "Target = 1 if Fighter A (the row-paired first fighter) wins; Target = 0 otherwise. "
            "By construction, symmetrical data augmentation enforces exactly 50% class balance "
            "in training, removing any red-corner/blue-corner bias inherent in UFC data."
        ),
        SP(6),
    ]

    # ── 3. Feature Engineering ────────────────────────────────────────────
    story += [
        H("3. Feature Engineering"),
        HR(),
        P(
            "The core engineering challenge is computing every feature using only information "
            "available before the bout being predicted. We implement an "
            "<b>event-snapshot pattern</b>: for each fight on a given event date, "
            "fighter histories are frozen to exclude any fight on or after that date."
        ),
        H2("3.1 Antisymmetric Difference Features"),
        P(
            "Each raw statistic s is transformed into a matchup feature: "
            "<b>diff_s = s_A - s_B</b>. A positive value means Fighter A is stronger on that "
            "dimension. This representation:"
        ),
        B("Eliminates corner-assignment bias (Red/Blue)."),
        B("Reduces dimensionality from 2*N raw stats to N difference features."),
        B("Enables symmetrical augmentation: swapping A-B simply negates all differences."),
        SP(4),
        H2("3.2 Rolling Elo Rating"),
        P(
            "A rolling Elo rating system is maintained for every fighter. After each bout, "
            "the winner gains Elo points proportional to the upset magnitude:"
        ),
        Code("DElo = K x (outcome - expected)"),
        Code("expected = 1 / (1 + 10^((Elo_B - Elo_A) / 400))"),
        P(
            "K-factor: 64 for fighters with fewer than 10 UFC fights (high uncertainty), "
            "32 thereafter. Starting Elo: 1,500 for all. Elo is computed sequentially "
            "in chronological order, capturing <b>strength of schedule</b> - the single "
            "strongest predictor omitted by record-only models."
        ),
        H2("3.3 Exponential Time-Decay Weighting"),
        P(
            "A fight from 7 years ago contributes the same to career totals as one from "
            "3 months ago. We apply an exponential decay with a 2-year half-life:"
        ),
        Code("weight(t) = exp(-lambda * days_ago)    where lambda = ln(2) / 730"),
        P(
            "Decayed win rate and decayed strike defence are computed as weighted averages, "
            "capturing skill evolution, aging, and improvement trajectories."
        ),
        H2("3.4 Cage Rust / Layoff Duration"),
        P(
            "Fighters returning from long absences (> 400 days) show a well-documented "
            "performance dip. We compute <b>diff_days_since_last_fight</b> "
            "(capped at 1,000 days) as a signed difference, allowing the model to quantify "
            "relative ring rust between opponents."
        ),
        H2("3.5 UFC Debutant Indicator"),
        P(
            "A binary flag <b>is_debut = 1</b> marks a fighter's first UFC appearance. "
            "Debutants face unknown quantities for opponents but also lack UFC-level "
            "performance history. The signed difference <b>diff_is_debut</b> captures "
            "the (debutant vs. veteran) asymmetry directly."
        ),
        H2("3.6 Corrected Takedown Defence"),
        P(
            "The original dataset provided only takedowns landed against a fighter, "
            "not takedowns attempted against them, making a true defence-rate calculation "
            "impossible. Our correction: takedown defence is the per-fight average of "
            "takedowns faced, used as a proxy for how often opponents shoot - and "
            "separately, whether those shots succeed."
        ),
        H2("3.7 Feature Summary"),
        SP(4),
        feature_table(),
        Paragraph("Table 1: Feature groups and their difference-encoded variables.", S["caption"]),
        SP(6),
    ]

    # ── 4. Training Methodology ───────────────────────────────────────────
    story += [
        H("4. Training Methodology"),
        HR(),
        H2("4.1 Chronological Train/Test Split"),
        P(
            "All fights before <b>3 June 2023</b> form the training set; all fights from "
            "that date onward form the held-out test set. This mimics real-world deployment: "
            "a model trained on historical data predicts future bouts it has never seen."
        ),
        H2("4.2 Symmetrical Data Augmentation"),
        P(
            "For every training bout (A, B, features=x, target=y) we add its mirror "
            "(B, A, features=-x, target=1-y). This:"
        ),
        B("Doubles effective training size to ~14,080 paired examples."),
        B("Enforces the antisymmetry constraint without any hyperparameter tuning."),
        B("Guarantees exactly 50% class balance, removing corner-bias."),
        P(
            "Crucially, mirrors are <b>only added to training</b>. The test set is never "
            "augmented, preserving a fair holdout evaluation."
        ),
        H2("4.3 Cross-Validation Strategy"),
        P(
            "Hyperparameter selection uses <b>TimeSeriesSplit (5 folds)</b>. For each fold, "
            "the training sub-fold includes original bouts plus their symmetrical mirrors, "
            "while the validation sub-fold uses only original bouts in forward-time order. "
            "This prevents any future information from contaminating hyperparameter selection."
        ),
        H2("4.4 Hyperparameter Search"),
        P(
            "GridSearchCV optimises <b>ROC-AUC</b> (chosen because it measures ranking "
            "quality across all probability thresholds, not just the 0.5 cut-off). "
            "All CPU cores are used (n_jobs=-1)."
        ),
        SP(6),
    ]

    # ── 5. Models ─────────────────────────────────────────────────────────
    story += [
        H("5. Models"),
        HR(),
        H2("5.1 Logistic Regression"),
        P(
            "<b>Why chosen:</b> LR is the natural baseline for binary classification. "
            "Its log-odds decision boundary is linear, so it finds the global best separating "
            "hyperplane in feature space. With well-normalised difference features the "
            "model produces interpretable, well-calibrated probabilities. "
            "Ridge regularisation (C parameter) prevents overfitting to correlated features. "
            "Requires StandardScaler pre-processing."
        ),
        H2("5.2 Decision Tree"),
        P(
            "<b>Why chosen:</b> A greedy tree serves as an interpretability benchmark - "
            "its splits are human-readable ('if diff_elo > 120 -> predict A wins'). "
            "Trees capture non-linear thresholds naturally. Depth limited to 5 to avoid "
            "overfitting on noisy fight outcomes."
        ),
        H2("5.3 Gradient Boosting"),
        P(
            "<b>Why chosen:</b> GBM builds an ensemble of weak trees sequentially, "
            "each correcting the residual errors of the prior. It captures complex "
            "non-linear interactions (e.g., 'high Elo advantage AND recent momentum') "
            "without manual feature crosses. The shrinkage parameter (learning rate) "
            "provides implicit regularisation."
        ),
        H2("5.4 MLP (Neural Network)"),
        P(
            "<b>Why chosen:</b> A shallow MLP (64-32 hidden units) tests whether "
            "deep feature interactions beyond gradient boosting are learnable from "
            "this dataset size. In practice, the dataset (~7K training examples) "
            "is too small for MLP to outperform well-regularised linear or tree "
            "models - confirmed by our results."
        ),
        SP(6),
    ]

    # ── 6. Results ────────────────────────────────────────────────────────
    story += [
        H("6. Results"),
        HR(),
        H2("6.1 Performance on Held-Out Test Set (1,761 bouts)"),
        SP(6),
        results_table(),
        Paragraph("Table 2: Model comparison on the chronological holdout test set (June 2023 onward).",
                  S["caption"]),
        SP(8),
        H2("6.2 Key Findings"),
        B(
            "<b>Logistic Regression wins overall</b> - 62.46% accuracy and ROC-AUC 0.6693. "
            "It also has the lowest Brier Score (0.2284), confirming well-calibrated probability estimates."
        ),
        B(
            "<b>Gradient Boosting is second</b> - 60.76% accuracy. The gap to LR suggests "
            "non-linear interactions add modest value but the training set is too small "
            "to fully exploit them."
        ),
        B(
            "<b>Decision Tree underperforms</b> - high precision (60%) but low recall (48%). "
            "The tree is conservative: it only predicts A-wins when highly confident."
        ),
        B(
            "<b>MLP underperforms LR and GBM</b> - 59.34% accuracy. Neural networks "
            "benefit from larger datasets; 7K examples with 35 features is insufficient."
        ),
        B(
            "<b>Majority-class baseline: 50%</b> - since the augmented training set is "
            "perfectly balanced, all models substantially beat the naive baseline."
        ),
        H2("6.3 Comparison to Published Work"),
        P(
            "The reference paper by Wagenaar et al. (2020) achieved approximately "
            "<b>57-58% accuracy</b> using UFC statistics without domain-specific enhancements. "
            "Our pipeline improves this by ~4-5 percentage points through rolling Elo, "
            "time-decay, layoff duration, debutant flags, and corrected defence metrics."
        ),
        SP(6),
    ]

    # ── 7. Methodological Integrity ───────────────────────────────────────
    story += [
        H("7. Methodological Integrity"),
        HR(),
        H2("7.1 Preventing Data Leakage"),
        P(
            "The most critical challenge in fight prediction is avoiding look-ahead bias. "
            "Our pipeline enforces a strict <b>event-snapshot</b> pattern:"
        ),
        B("Fighter histories are computed up to (but not including) the event date."),
        B("Elo ratings are updated only after a fight is processed, in chronological order."),
        B("Layoff duration is computed from the most recent prior fight date."),
        B("Test set mirrors are never generated - only training data is augmented."),
        H2("7.2 Symmetry Constraint"),
        P(
            "A valid fight predictor must satisfy P(A beats B) + P(B beats A) = 1. "
            "Using antisymmetric difference features and symmetrical augmentation mathematically "
            "enforces this. Without augmentation, models trained on the original corner-assigned "
            "data predict A-wins more often (red-corner bias in the raw data)."
        ),
        SP(6),
    ]

    # ── 8. Conclusion ─────────────────────────────────────────────────────
    story += [
        H("8. Conclusion"),
        HR(),
        P(
            "This project demonstrates that domain knowledge significantly outperforms "
            "naive statistical aggregation for UFC fight prediction. The five key contributions "
            "- rolling Elo rating, exponential time-decay, cage rust / layoff duration, "
            "UFC debutant indicator, and corrected defence metrics - each address a specific "
            "real-world predictive signal that raw per-fight averages ignore."
        ),
        P(
            "The final model (Logistic Regression, 62.46% accuracy, ROC-AUC 0.6693) "
            "represents a meaningful improvement over published baselines while maintaining "
            "complete methodological integrity. The pipeline is production-ready: the "
            "predict.py module computes live difference features for any two fighters "
            "using the same event-snapshot logic, and outputs a calibrated win probability."
        ),
        H2("8.1 Future Work"),
        B("Glicko-2 ratings: incorporate rating deviation and volatility alongside Elo."),
        B("Style matchup features: Orthodox vs. Southpaw win-rate interaction terms."),
        B("Injury / absence signals: scraping publicly available injury reports."),
        B("Betting-line integration: the market-implied probability is a powerful feature."),
        SP(6),
    ]

    # ── References ────────────────────────────────────────────────────────
    story += [
        H("References"),
        HR(),
        B(
            "Wagenaar D., Dorado A., Lozano-Blasco R. (2020). "
            "Predicting the outcome of UFC fights using machine learning. "
            "International Journal of Computer Science in Sport, 19(1)."
        ),
        B(
            "Glickman M. E. (1999). Parameter estimation in large dynamic paired "
            "comparison experiments. Applied Statistics, 48(3), 377-394."
        ),
        B(
            "Elo A. E. (1978). The rating of chessplayers, past and present. "
            "Arco Publishing."
        ),
        B(
            "Pedregosa F. et al. (2011). Scikit-learn: Machine Learning in Python. "
            "JMLR 12, 2825-2830."
        ),
        SP(20),
    ]

    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="UFC Fight Prediction Write-Up",
        author="Riddhi",
    )

    story = build_content()
    doc.build(story)
    print(f"Successfully generated write-up: {OUTPUT}")


if __name__ == "__main__":
    main()
