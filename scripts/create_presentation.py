import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Output file
OUTPUT_PPTX = Path("UFC_Fight_Prediction_Presentation.pptx")
FIGURE_DIR = Path("results/figures")

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# Theme Colors
BG_COLOR = RGBColor(11, 15, 25)         # Deep slate/black #0B0F19
CARD_BG = RGBColor(24, 32, 47)          # Card navy #18202F
CARD_BORDER = RGBColor(51, 65, 85)      # Slate border #334155
ACCENT_RED = RGBColor(225, 29, 72)      # UFC Crimson #E11D48
ACCENT_GOLD = RGBColor(245, 158, 11)    # Gold Amber #F59E0B
ACCENT_CYAN = RGBColor(56, 189, 248)    # Sky Cyan #38BDF8
ACCENT_GREEN = RGBColor(16, 185, 129)   # Emerald Green #10B981
TEXT_WHITE = RGBColor(248, 250, 252)    # Pure White #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184)    # Slate Muted #94A3B8

def set_slide_background(slide):
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_COLOR
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, title, category="UE24CS352A · MACHINE LEARNING MINI-PROJECT · STATEMENT 74"):
    # Category tag
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = category.upper()
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_RED
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

def add_card(slide, left, top, width, height, title=None, border_color=CARD_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)
    
    if title:
        title_box = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
        p = title_box.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN
    return card

# -------------------------------------------------------------
# SLIDE 1: TITLE SLIDE
# -------------------------------------------------------------
s1 = prs.slides.add_slide(blank_layout)
set_slide_background(s1)

# Large Title Card
add_card(s1, Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.1), border_color=ACCENT_RED)

tbox = s1.shapes.add_textbox(Inches(1.5), Inches(1.6), Inches(10.3), Inches(4.3))
tf = tbox.text_frame
tf.word_wrap = True

p1 = tf.paragraphs[0]
p1.text = "UE24CS352A — MACHINE LEARNING MINI-PROJECT"
p1.font.size = Pt(13)
p1.font.bold = True
p1.font.color.rgb = ACCENT_GOLD

p2 = tf.add_paragraph()
p2.text = "Applying Machine Learning Algorithms to Predict UFC Fight Outcomes"
p2.font.size = Pt(32)
p2.font.bold = True
p2.font.color.rgb = TEXT_WHITE
p2.space_before = Pt(14)
p2.space_after = Pt(14)

p3 = tf.add_paragraph()
p3.text = "Problem Statement 74  |  Binary Matchup Classification  |  Chronological Holdout (8,755 Bouts)"
p3.font.size = Pt(15)
p3.font.color.rgb = ACCENT_CYAN

p4 = tf.add_paragraph()
p4.text = "\nTeam: 2 Members  •  Repository: UFC-Fight-Predictions  •  Semester 5"
p4.font.size = Pt(14)
p4.font.color.rgb = TEXT_MUTED

# -------------------------------------------------------------
# SLIDE 2: PROBLEM STATEMENT & DOMAIN REALITY
# -------------------------------------------------------------
s2 = prs.slides.add_slide(blank_layout)
set_slide_background(s2)
add_header(s2, "Problem Statement: The Challenge of Combat Sports Prediction")

# Left Card: Problem Formulation
add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "Problem Formulation")
box_l = s2.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
tf_l = box_l.text_frame
tf_l.word_wrap = True

bullets_l = [
    ("Goal:", " Given two fighters scheduled to compete, predict the winner using strictly information available before fight night."),
    ("Pairwise Classification:", " The fundamental ML challenge is modeling a matchup between two entities rather than a single individual."),
    ("Antisymmetric Properties:", " If Fighter A vs Fighter B produces feature vector x, swapping the fighters must produce -x and inverse label."),
    ("Strict Pre-Fight Horizon:", " No lifetime stats, post-fight metrics, or future bout records may contaminate past predictions."),
]
for title, body in bullets_l:
    p = tf_l.add_paragraph()
    run_t = p.add_run()
    run_t.text = title
    run_t.font.bold = True
    run_t.font.color.rgb = ACCENT_GOLD
    run_b = p.add_run()
    run_b.text = body
    run_b.font.color.rgb = TEXT_WHITE
    p.font.size = Pt(13)
    p.space_after = Pt(12)

# Right Card: The Domain Ceiling
add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "The Combat Sports Reality & Hard Ceiling")
box_r = s2.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
tf_r = box_r.text_frame
tf_r.word_wrap = True

bullets_r = [
    ("Low Signal-to-Noise Ratio:", " MMA features extreme volatility: a single 'puncher's chance' knockout can invalidate lifetime statistical superiority."),
    ("The Academic Ceiling:", " Prior literature (Hitkul et al., McQuaide) reports accuracies between 51.7% and 60.2%. Any model claiming 75-80% suffers from hidden data leakage."),
    ("Corner Bias Pitfall:", " UFCStats lists the higher-ranked / favourite fighter as the Red Corner (62.6% lifetime win rate). Naive corner prediction hides true model efficacy."),
    ("Real-World Objective:", " A calibrated model that beats the coin-flip baseline by 10+ points and identifies high-confidence bouts is practically informative.")
]
for title, body in bullets_r:
    p = tf_r.add_paragraph()
    run_t = p.add_run()
    run_t.text = title
    run_t.font.bold = True
    run_t.font.color.rgb = ACCENT_RED
    run_b = p.add_run()
    run_b.text = body
    run_b.font.color.rgb = TEXT_WHITE
    p.font.size = Pt(13)
    p.space_after = Pt(12)

# -------------------------------------------------------------
# SLIDE 3: REFERENCE STUDY CRITIQUE & LESSONS
# -------------------------------------------------------------
s3 = prs.slides.add_slide(blank_layout)
set_slide_background(s3)
add_header(s3, "Reference Study Review: Identifying Core Methodological Flaws")

add_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "McKinley McQuaide (Stanford CS229, 2019)")
box3_l = s3.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
tf3_l = box3_l.text_frame
tf3_l.word_wrap = True

bullets3_l = [
    ("Reported Headline:", " Reported ~60% test accuracy across SGD Perceptron, Decision Trees, MLPs, and Gradient Boosting on 3,355 fights."),
    ("The Red-Corner Trap:", " Across the study dataset, the Red corner won 62.6% of the time. Trivial dummy guessing of 'Red' outperformed all 4 trained models!"),
    ("Problem Formulation Flaw:", " Evaluated multi-class (win/draw/loss) using Mean Squared Error (MSE), which is mathematically unsuited for categorical classification."),
    ("Massive Overfitting:", " MLP train accuracy reached 89.6% vs test 57.7%; Gradient Boosting train 88.6% vs test 61.2% due to 134 unregularized features.")
]
for title, body in bullets3_l:
    p = tf3_l.add_paragraph()
    run_t = p.add_run()
    run_t.text = title
    run_t.font.bold = True
    run_t.font.color.rgb = ACCENT_RED
    run_b = p.add_run()
    run_b.text = body
    run_b.font.color.rgb = TEXT_WHITE
    p.font.size = Pt(13)
    p.space_after = Pt(12)

add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Our Architectural Corrections")
box3_r = s3.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
tf3_r = box3_r.text_frame
tf3_r.word_wrap = True

bullets3_r = [
    ("Unbiased Corner Formulation:", " Swapped Red/Blue for neutral Fighter A vs Fighter B with signed difference features. The majority baseline is exactly 50.71%."),
    ("Binary Target Cleaning:", " Discarded draws and no-contests (<1.5% of UFC fights) to focus strictly on decisive binary victory prediction."),
    ("Strict Chronological Holdout:", " 6,994 fights (1994-2023) train; 1,761 fights (June 2023-Sept 2026) untouched test holdout, validated via TimeSeriesSplit."),
    ("High Regularization & Parsimony:", " Pruned 134 noisy variables down to 35 domain-justified difference features, enforcing strict Occam's razor.")
]
for title, body in bullets3_r:
    p = tf3_r.add_paragraph()
    run_t = p.add_run()
    run_t.text = title
    run_t.font.bold = True
    run_t.font.color.rgb = ACCENT_GREEN
    run_b = p.add_run()
    run_b.text = body
    run_b.font.color.rgb = TEXT_WHITE
    p.font.size = Pt(13)
    p.space_after = Pt(12)

# -------------------------------------------------------------
# SLIDE 4: THE HERO DISCOVERY - LABEL SCRAMBLING BUG
# -------------------------------------------------------------
s4 = prs.slides.add_slide(blank_layout)
set_slide_background(s4)
add_header(s4, "Diagnostic Breakthrough: The Hidden UFCStats Label Inversion")

add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3), "A Critical Diagnostic Milestone That Rescued The Pipeline", border_color=ACCENT_RED)
box4 = s4.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(11.1), Inches(4.5))
tf4 = box4.text_frame
tf4.word_wrap = True

narrative = [
    ("1. The Silent Defect:", " UFCStats states fight OUTCOME relative to the bout string ('Fighter A vs. Fighter B'). However, per-round fighter statistics are read from a separate table whose row order matched the bout title only 49.4% of the time — essentially a coin flip!"),
    ("2. The False Sense of Progress:", " Taking OUTCOME at face value scrambled half the dataset's labels. Initial models reported 56% headline accuracy, which seemed plausible, but every model had actually collapsed into a single majority class."),
    ("3. The Diagnostic Catch:", " The defect was completely hidden by headline accuracy. What exposed it was a test Recall of exactly 1.000 and ROC-AUC of 0.519 (pure random noise). This diagnostic rigor prevented months of training on corrupted data."),
    ("4. The Resolution:", " We re-engineered build_fight_dataset.py to resolve fight winners by exact string matching against both sides of the bout title. This single fix repaired the ground truth target AND six accumulating career history features (wins, losses, win rates)."),
    ("5. The Impact:", " Resolving labels immediately boosted Logistic Regression accuracy from 56% (random) to 61.1% and ROC-AUC from 0.519 to 0.647!")
]
for title, body in narrative:
    p = tf4.add_paragraph()
    run_t = p.add_run()
    run_t.text = title
    run_t.font.bold = True
    run_t.font.color.rgb = ACCENT_GOLD
    run_b = p.add_run()
    run_b.text = body
    run_b.font.color.rgb = TEXT_WHITE
    p.font.size = Pt(13)
    p.space_after = Pt(9)

# -------------------------------------------------------------
# SLIDE 5: FEATURE ENGINEERING & DOMAIN MODELING
# -------------------------------------------------------------
s5 = prs.slides.add_slide(blank_layout)
set_slide_background(s5)
add_header(s5, "Domain Feature Engineering: 35 Leakage-Free Matchup Features")

# Card 1: Elo & Recency Decay
add_card(s5, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.3), "Opponent Quality & Recency")
b5_1 = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.3), Inches(4.5))
tf5_1 = b5_1.text_frame
tf5_1.word_wrap = True
items5_1 = [
    ("Rolling Pre-Fight Elo:", " Base 1500; dynamic K=64 for first 3 fights, K=32 thereafter. 1.2x multiplier for KO/Sub, 0.8x for split decisions. Captures strength of schedule."),
    ("Exponential Time Decay:", " Half-life of 2 years (730 days). Weights older fights by 2^(-days/730). Models fighter physical aging & skill evolution."),
    ("Decayed Win Rate & Defense:", " Recency-weighted win rate and striking defense provide smooth capability curves.")
]
for t, b in items5_1:
    p = tf5_1.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_CYAN
    p.add_run().text = b
    p.font.size = Pt(11)
    p.space_after = Pt(8)

# Card 2: Combat Dynamics
add_card(s5, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.3), "Combat Dynamics & Rest")
b5_2 = s5.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.5))
tf5_2 = b5_2.text_frame
tf5_2.word_wrap = True
items5_2 = [
    ("Cage Rust / Layoff Duration:", " Days since previous bout (capped at 1000). Test data confirms a -0.086 negative correlation with winning for long layoffs."),
    ("UFC Debutant Factor:", " Binary indicator separating Octagon debutants (0 UFC fights) from cage veterans (who win ~62% of debut bouts)."),
    ("Corrected Takedown Defense:", " Fixed calculation: 1.0 - (td_absorbed / td_faced). Normalizes defensive wrestling correctly."),
    ("Significant Strike Defense:", " Tracks pace-normalized defense: 1.0 - (sig_absorbed / sig_faced).")
]
for t, b in items5_2:
    p = tf5_2.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_GOLD
    p.add_run().text = b
    p.font.size = Pt(11)
    p.space_after = Pt(8)

# Card 3: Physical & Differences
add_card(s5, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3), "Physical & Matchup Differences")
b5_3 = s5.shapes.add_textbox(Inches(9.0), Inches(2.2), Inches(3.3), Inches(4.5))
tf5_3 = b5_3.text_frame
tf5_3.word_wrap = True
items5_3 = [
    ("Physical Differences:", " Age (calculated precisely on fight night), height, reach, and southpaw stance differences."),
    ("Missing Value Flags:", " Missing reach/height paired with _known binary indicators, allowing models to separate 'equal' from 'unknown'."),
    ("Antisymmetric Design:", " Every genuine feature is diff = Fighter A - Fighter B. Swapping the order flips the sign identically."),
    ("Event Snapshotting:", " Stats are updated strictly AFTER all fights on a card conclude. Zero within-card leakage.")
]
for t, b in items5_3:
    p = tf5_3.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_GREEN
    p.add_run().text = b
    p.font.size = Pt(11)
    p.space_after = Pt(8)

# -------------------------------------------------------------
# SLIDE 6: SYMMETRICAL DATA AUGMENTATION & VALIDATION
# -------------------------------------------------------------
s6 = prs.slides.add_slide(blank_layout)
set_slide_background(s6)
add_header(s6, "Methodology: Symmetrical Augmentation & Leakage-Free Validation")

add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "Symmetrical Data Augmentation")
b6_l = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
tf6_l = b6_l.text_frame
tf6_l.word_wrap = True
bullets6_l = [
    ("Mathematical Law:", " If Fighter A vs B has feature vector x and label 1, then B vs A MUST have vector -x (for antisymmetric features) and label 0."),
    ("Mirroring Training Bouts:", " Each training fight is duplicated as (B, A, -x, 0), doubling training size from 6,994 to 13,988 bouts."),
    ("Zero Corner / Order Bias:", " Forces models to learn antisymmetric boundaries: P(A beats B) = 1 - P(B beats A) exactly."),
    ("Perfect Class Balance:", " Symmetrical pairing guarantees an exact 50.00% label balance and zero-mean difference vectors.")
]
for t, b in bullets6_l:
    p = tf6_l.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_CYAN
    p.add_run().text = b
    p.font.size = Pt(13)
    p.space_after = Pt(12)

add_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Zero-Leakage TimeSeriesSplit Validation")
b6_r = s6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
tf6_r = b6_r.text_frame
tf6_r.word_wrap = True
bullets6_r = [
    ("Why KFold Fails:", " Standard KFold permits training on future bouts to predict past bouts, violating the causal arrow of time."),
    ("Chronological Split:", " 6,994 fights (1994 to May 2023) for training; 1,761 fights (June 2023 to Sept 2026) untouched test holdout."),
    ("Symmetrical TimeSeriesSplit CV:", " 5-fold TimeSeriesSplit splits original chronological bouts. Training folds receive symmetrical mirrors, while validation folds remain untouched forward bouts (0 leakage)."),
    ("Holdout Independence:", " Scalers and transformers are fitted strictly on training data; the 1,761 test fights represent true future deployment.")
]
for t, b in bullets6_r:
    p = tf6_r.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_GOLD
    p.add_run().text = b
    p.font.size = Pt(13)
    p.space_after = Pt(12)

# -------------------------------------------------------------
# SLIDE 7: MODEL ARCHITECTURES & TUNING RATIONALE
# -------------------------------------------------------------
s7 = prs.slides.add_slide(blank_layout)
set_slide_background(s7)
add_header(s7, "Model Selection: Four Families Across The Complexity Spectrum")

models_info = [
    ("Logistic Regression", "Strongest Model (Winner)", "Regularized linear classifier with StandardScaler. L2 penalty (C=0.05). Low variance, strong global calibration, and naturally obeys antisymmetry. High resilience to fight noise.", ACCENT_GREEN),
    ("Gradient Boosting", "Non-linear Ensemble", "GradientBoostingClassifier (200 trees, depth 2, lr 0.05). Captures non-linear feature interactions (e.g. steep age decline cliff).", ACCENT_CYAN),
    ("Decision Tree", "Interpretable Baseline", "Cost-complexity pruned tree (depth 5, min leaf 25). Identifies dominant threshold rules (age diff, Elo splits).", ACCENT_GOLD),
    ("Multilayer Perceptron", "Neural Architecture", "MLPClassifier (16 neurons, alpha=0.1, Adam). Explores continuous latent representations, but struggles with combat noise variance.", ACCENT_RED),
]

for i, (name, role, desc, col) in enumerate(models_info):
    col_idx = i % 2
    row_idx = i // 2
    x = Inches(0.8 + col_idx * 6.0)
    y = Inches(1.6 + row_idx * 2.7)
    add_card(s7, x, y, Inches(5.7), Inches(2.4), name, border_color=col)
    
    tb = s7.shapes.add_textbox(x + Inches(0.2), y + Inches(0.6), Inches(5.3), Inches(1.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p1 = tf.paragraphs[0]
    p1.text = role.upper()
    p1.font.size = Pt(11)
    p1.font.bold = True
    p1.font.color.rgb = col
    
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(12)
    p2.font.color.rgb = TEXT_WHITE
    p2.space_before = Pt(4)

# -------------------------------------------------------------
# SLIDE 8: RESULTS, BENCHMARKS & HIGH CONFIDENCE SCALING
# -------------------------------------------------------------
s8 = prs.slides.add_slide(blank_layout)
set_slide_background(s8)
add_header(s8, "Results: Exceeding Literature Benchmarks (1,761 Test Fights)")

# Left Card: Comparison Table
add_card(s8, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3), "Model Performance Comparison")
tb8 = s8.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.6), Inches(4.5))
tf8 = tb8.text_frame
tf8.word_wrap = True

p_base = tf8.paragraphs[0]
p_base.text = "Majority-Class Baseline: 50.71%  |  Stanford Ref Study: ~60.0%"
p_base.font.size = Pt(12)
p_base.font.bold = True
p_base.font.color.rgb = ACCENT_GOLD
p_base.space_after = Pt(10)

table_data = [
    ("Model", "Accuracy", "ROC-AUC", "F1", "Brier"),
    ("Logistic Regression", "62.46%", "0.6693", "0.6293", "0.2284"),
    ("Gradient Boosting", "60.76%", "0.6488", "0.6107", "0.2330"),
    ("MLP (Neural Net)", "59.34%", "0.6243", "0.5707", "0.2494"),
    ("Decision Tree", "58.66%", "0.6217", "0.5333", "0.2433"),
]

for row in table_data[1:]:
    p = tf8.add_paragraph()
    is_lr = "Logistic" in row[0]
    p.text = f"{row[0]:<20}  Acc: {row[1]}  AUC: {row[2]}  Brier: {row[4]}"
    p.font.size = Pt(12)
    p.font.bold = is_lr
    p.font.color.rgb = ACCENT_GREEN if is_lr else TEXT_WHITE
    p.space_after = Pt(6)

p_extra = tf8.add_paragraph()
p_extra.text = "\nKey Takeaways:\n• Logistic Regression wins across all metrics (+11.8% over baseline).\n• Brier score of 0.228 beats random coin-flip (0.250).\n• Heavily regularized, simpler architectures beat complex networks."
p_extra.font.size = Pt(11)
p_extra.font.color.rgb = TEXT_MUTED

# Right Card: Confidence Tiers & Calibration
add_card(s8, Inches(7.0), Inches(1.6), Inches(5.5), Inches(5.3), "Accuracy Scales Monotonically With Confidence")
tb8_r = s8.shapes.add_textbox(Inches(7.2), Inches(2.2), Inches(5.1), Inches(4.5))
tf8_r = tb8_r.text_frame
tf8_r.word_wrap = True

p_c = tf8_r.paragraphs[0]
p_c.text = "Model is right more often when it is more sure:"
p_c.font.size = Pt(13)
p_c.font.color.rgb = TEXT_WHITE
p_c.space_after = Pt(8)

tiers = [
    ("Lowest (0-20%):", "52.7% accuracy  (Coin-flip range)", TEXT_MUTED),
    ("Low (20-40%):", "57.4% accuracy", TEXT_MUTED),
    ("Medium (40-60%):", "58.8% accuracy", TEXT_MUTED),
    ("High (60-80%):", "70.2% accuracy", ACCENT_CYAN),
    ("Highest (Top 20%):", "73.3% accuracy  (258 / 352 correct!)", ACCENT_GREEN),
]
for t, val, col in tiers:
    p = tf8_r.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = col
    p.add_run().text = val
    p.font.size = Pt(12)
    p.space_after = Pt(6)

p_proof = tf8_r.add_paragraph()
p_proof.text = "\nProof of Real Signal:\nA model fitting noise produces a flat confidence-accuracy line. Reaching 73.3% on confident bouts proves genuine predictive structure was learned."
p_proof.font.size = Pt(11)
p_proof.font.color.rgb = ACCENT_GOLD

# -------------------------------------------------------------
# SLIDE 9: FEATURE IMPORTANCE & WHAT THE MODEL LEARNED
# -------------------------------------------------------------
s9 = prs.slides.add_slide(blank_layout)
set_slide_background(s9)
add_header(s9, "Model Interpretability: Physical & Tactical Insights")

add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "Top Logistic Regression Drivers")
tb9_l = s9.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
tf9_l = tb9_l.text_frame
tf9_l.word_wrap = True

coeffs = [
    ("+0.461", "diff_elo", "Superior pre-fight rating & strength of schedule"),
    ("+0.322", "diff_decayed_sig_str_defence", "Higher recent strike defense percentage"),
    ("-0.296", "diff_age", "Younger fighters win consistently in modern UFC"),
    ("-0.277", "diff_sig_str_absorbed_per_fight", "Fighters absorbing heavy damage lose more often"),
    ("+0.273", "diff_decayed_win_rate", "Recency-weighted momentum and recent form"),
    ("+0.199", "diff_td_attempted_per_fight", "Proactive offensive wrestling threat"),
]
for val, name, desc in coeffs:
    p = tf9_l.add_paragraph()
    col = ACCENT_GREEN if val.startswith("+") else ACCENT_RED
    r_val = p.add_run()
    r_val.text = f"{val} "
    r_val.font.bold = True
    r_val.font.color.rgb = col
    r_name = p.add_run()
    r_name.text = f"{name}: "
    r_name.font.bold = True
    r_name.font.color.rgb = TEXT_WHITE
    r_desc = p.add_run()
    r_desc.text = desc
    r_desc.font.color.rgb = TEXT_MUTED
    p.font.size = Pt(11)
    p.space_after = Pt(8)

add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Tactical & Division Insights")
tb9_r = s9.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
tf9_r = tb9_r.text_frame
tf9_r.word_wrap = True

insights = [
    ("Independent Agreement:", " Both Gradient Boosting and Logistic Regression independently ranked Elo, Age, and Strike Defense as the top three factors."),
    ("Division Variance:", " Accuracy peaks at 68.8% in Women's Bantamweight and 68.1% in Middleweight, but drops to 50.9% in Light Heavyweight — a division defined by 1-punch knockout volatility."),
    ("The Aging Factor:", " Reproduces McQuaide's primary finding: older fighters suffer an accelerating physiological disadvantage."),
    ("Rest & Inactivity:", " Cage rust (days since last fight) has a statistically negative coefficient, confirming that ring rust is real.")
]
for t, b in insights:
    p = tf9_r.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_GOLD
    p.add_run().text = b
    p.font.size = Pt(12)
    p.space_after = Pt(10)

# -------------------------------------------------------------
# SLIDE 10: LIVE DEMO, CONCLUSIONS & FUTURE WORK
# -------------------------------------------------------------
s10 = prs.slides.add_slide(blank_layout)
set_slide_background(s10)
add_header(s10, "Conclusions & Live Demonstration")

add_card(s10, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3), "Live Demonstration (src/predict.py)")
tb10_l = s10.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.5))
tf10_l = tb10_l.text_frame
tf10_l.word_wrap = True

bullets10_l = [
    ("Live CLI Scoring:", " python src/predict.py 'Islam Makhachev' 'Charles Oliveira'"),
    ("Predicted Probabilities:", " Makhachev: 67.9%  |  Oliveira: 32.1%"),
    ("Key Matchup Drivers:", " Elo (+58.4), Decayed Win Rate (+0.25), Strike Defense (+13%), Takedown Defense (+39%), Layoff (-161 days)."),
    ("Symmetry Verification:", " Reversing fighter order produces EXACT inversion (Oliveira 32.1%, Makhachev 67.9%), proving zero order bias live.")
]
for t, b in bullets10_l:
    p = tf10_l.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_CYAN
    p.add_run().text = b
    p.font.size = Pt(12)
    p.space_after = Pt(10)

add_card(s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Core Conclusions & Future Work")
tb10_r = s10.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.5))
tf10_r = tb10_r.text_frame
tf10_r.word_wrap = True

bullets10_r = [
    ("Label Provenance is Paramount:", " Ground truth labels deserve as much scrutiny as features. Diagnosing label inversion was the project's defining breakthrough."),
    ("Symmetry by Construction:", " Pairwise matchup problems require antisymmetric difference features and symmetrical data augmentation."),
    ("High-Confidence Utility:", " Achieving 73.3% in top-tier confidence shows that knowing WHEN the model knows is more valuable than an uncalibrated headline number."),
    ("Future Extensions:", " Method-of-victory multi-task modeling (KO vs Sub vs Dec), division-specific chin/power ratings, and betting market EV backtesting.")
]
for t, b in bullets10_r:
    p = tf10_r.add_paragraph()
    p.add_run().text = t + " "
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = ACCENT_GOLD
    p.add_run().text = b
    p.font.size = Pt(12)
    p.space_after = Pt(10)

# Save
prs.save(OUTPUT_PPTX)
print(f"Successfully generated presentation: {OUTPUT_PPTX}")
