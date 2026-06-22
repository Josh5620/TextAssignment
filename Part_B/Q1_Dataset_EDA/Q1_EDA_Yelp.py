"""
Part B - Q1: Dataset and Exploratory Data Analysis (EDA)  [Owner: Ying Xin]

Dataset : Yelp Review Full (balanced 30k sample, 6,000 reviews per star 1..5),
          remapped to 3-class sentiment (negative = 1-2 stars, neutral = 3,
          positive = 4-5). Collapsing the balanced 5-star sample yields a natural
          2:1:2 class distribution (negative:neutral:positive) which the EDA shows.
Problem : Supervised 3-class SENTIMENT classification - predict whether a customer
          review is negative / neutral / positive from its raw text. The neutral
          class is the genuinely hard one (it overlaps both poles), which is what
          makes the 4-model comparison diverge most (see FINDINGS.md spread test).

This script performs the full EDA and all data-preparation activities required
for the 5-mark EDA component, and saves the cleaned dataset that the Q2/Q3/Q4
teammates will load. Charts are written to the eda_outputs/ folder for the report.

Run:  python Q1_EDA_Yelp.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless backend: save figures, do not pop up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from clean_review import clean_review   # standard text-cleaning (no neg_/emoji feature-engineering)


# ---- paths ----
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
RAW_FILE = DATA_DIR / "yelp_review_full_raw_30k.csv"
CLEAN_FILE = DATA_DIR / "yelp_clean.csv"
OUT_DIR = HERE / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

TEXT_COL = "review"
LABEL_COL = "rating"      # raw star rating (1..5), kept in the saved dataset
SENT_COL = "sentiment"    # 3-class target derived from rating: negative/neutral/positive


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def to_sentiment(rating):
    # Collapse the 1..5 star rating into 3 sentiment classes.
    # 1-2 -> negative, 3 -> neutral, 4-5 -> positive.
    return np.where(rating <= 2, "negative", np.where(rating == 3, "neutral", "positive"))


def chi2_top_words_per_class(texts, labels, top_n=12):
    """Chi-square test: words most associated with each sentiment class.

    For each class (negative/neutral/positive) we run a one-vs-rest chi-square over
    TF-IDF features (same TF-IDF config as the models, so the EDA matches the
    pipeline). We then keep only words that are OVER-represented in that class
    (mean TF-IDF inside the class > outside), so 'negative' surfaces genuinely
    angry words and 'positive' happy ones, and rank those by chi-square score.
    Returns {class: [(word, score), ...]} ordered negative -> neutral -> positive.
    """
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    features = np.array(vec.get_feature_names_out())

    result = {}
    for label in sorted(labels.unique()):     # negative < neutral < positive (alphabetical = neg->pos)
        mask = (labels == label).values
        target = mask.astype(int)
        scores, _ = chi2(X, target)

        # direction: keep only words more frequent INSIDE this class than outside
        mean_in = np.asarray(X[mask].mean(axis=0)).ravel()
        mean_out = np.asarray(X[~mask].mean(axis=0)).ravel()
        characteristic = mean_in > mean_out

        order = np.argsort(scores)[::-1]
        picks = [i for i in order if characteristic[i]][:top_n]
        result[label] = [(features[i], float(scores[i])) for i in picks]
    return result


def main():
    # ---------- 1. Load ----------
    df = pd.read_csv(RAW_FILE)
    df[SENT_COL] = to_sentiment(df[LABEL_COL])   # 3-class target derived from stars

    section("1. Dataset Overview")
    print(f"Source file : {RAW_FILE.name}")
    print(f"Rows        : {df.shape[0]}")
    print(f"Columns     : {list(df.columns)}")
    print("\nSample records:")
    print(df.head(3).to_string())

    # ---------- 2. Data quality ----------
    section("2. Missing Values")
    print(df[[TEXT_COL, LABEL_COL]].isnull().sum())

    section("3. Duplicate Reviews")
    print(f"Exact duplicate review texts: {df.duplicated(subset=[TEXT_COL]).sum()}")

    # ---------- 3. Class distribution ----------
    # The raw stars are balanced (6k each), but collapsing to 3 sentiment classes
    # gives a natural 2:1:2 (negative:neutral:positive) imbalance -- we show it
    # here because it motivates using macro-averaged metrics in Q4.
    section("4. Class Distribution (3-class sentiment)")
    order = ["negative", "neutral", "positive"]
    dist = df[SENT_COL].value_counts().reindex(order)
    print(dist)
    print("(Underlying stars are balanced 6k each; the 2:1:2 split comes from the")
    print(" 1-2 / 3 / 4-5 collapse -> handle with class_weight + macro metrics in Q4.)")

    plt.figure(figsize=(7, 4))
    plt.bar(order, dist.values, color=["#C0392B", "#C9B037", "#2E8B57"])
    plt.title("Yelp review count per sentiment class (3-class)")
    plt.xlabel("Sentiment class")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=120)
    plt.close()

    # ---------- 4. Clean (all data-preparation happens here) ----------
    section("5. Data Preparation / Cleaning")
    print("Applying clean_review (lowercase, strip URLs/HTML, expand contractions,")
    print("keep negation words, remove emoticons, remove stopwords, lemmatize)...")
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df["clean_text"] = df[TEXT_COL].apply(clean_review)
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""].copy()
    print(f"Rows before cleaning: {before}")
    print(f"Rows after  cleaning: {len(df)}  (empty rows dropped: {before - len(df)})")

    # ---------- 5. Text length analysis (before vs after) ----------
    df["char_count"] = df[TEXT_COL].astype(str).str.len()
    df["words_before"] = df[TEXT_COL].astype(str).str.split().str.len()
    df["words_after"] = df["clean_text"].str.split().str.len()

    section("6. Text Length Summary")
    print(df[["char_count", "words_before", "words_after"]].describe().round(1))

    section("7. Average Cleaned Word Count by Sentiment Class")
    print(df.groupby(SENT_COL)["words_after"].mean().round(1).reindex(["negative", "neutral", "positive"]))

    plt.figure(figsize=(8, 4))
    plt.hist(df["words_before"], bins=50, color="#C44E52", alpha=0.7, label="before cleaning")
    plt.hist(df["words_after"], bins=50, color="#4C72B0", alpha=0.7, label="after cleaning")
    plt.title("Review length distribution (words)")
    plt.xlabel("Words per review")
    plt.ylabel("Number of reviews")
    plt.xlim(0, 400)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "length_distribution.png", dpi=120)
    plt.close()

    # ---------- 6. Chi-square: most distinctive words per sentiment class ----------
    # (Replaces the old overall top-words bar chart and the pos/neg word clouds,
    #  which were frequency-based and dominated by shared neutral nouns. Chi-square
    #  shows the words that actually SEPARATE the sentiment classes -- and visually
    #  proves the neutral class overlaps both poles = "not trivially separable".)
    section("8. Chi-square: most distinctive words per sentiment class (negative .. positive)")
    chi_words = chi2_top_words_per_class(df["clean_text"], df[SENT_COL])
    for label in sorted(chi_words):
        words = ", ".join(w for w, _ in chi_words[label])
        print(f"{label}: {words}")

    # negative (red) -> neutral (amber) -> positive (green) spectrum, one per panel
    spectrum = ["#C0392B", "#C9B037", "#2E8B57"]
    n_classes = len(chi_words)
    fig, axes = plt.subplots(n_classes, 1, figsize=(9, 2.6 * n_classes))
    for ax, label, color in zip(axes, sorted(chi_words), spectrum):
        pairs = chi_words[label][::-1]           # smallest at bottom, largest on top
        ax.barh([w for w, _ in pairs], [s for _, s in pairs], color=color)
        ax.set_title(f"{label} — most distinctive words (chi-square)", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.supxlabel("chi-square score (word vs this sentiment class)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chi2_top_words_per_class.png", dpi=120)
    plt.close(fig)
    print("Saved chi-square chart -> chi2_top_words_per_class.png")

    # ---------- 7. Save cleaned dataset for teammates (Q2/Q3/Q4) ----------
    # Primary label is 'sentiment' (3-class). Raw 'rating' (1..5) is kept too so a
    # binary (neg vs pos) or 5-class variant can still be derived later if needed.
    out = df[[TEXT_COL, "clean_text", LABEL_COL, SENT_COL]].copy()
    out.to_csv(CLEAN_FILE, index=False)
    section("9. Saved Clean Dataset")
    print(f"Saved -> {CLEAN_FILE}")
    print(f"Columns: {list(out.columns)}  (label = '{SENT_COL}': negative/neutral/positive; '{LABEL_COL}' kept raw)")
    print(f"Rows   : {len(out)}")

    # ---------- 8. Proposed predictive models (one per member) ----------
    section("10. Proposed Predictive Models (4 members)")
    proposals = [
        "Logistic Regression - strong, fast linear baseline for sparse TF-IDF text; "
        "weighs all word features together, well suited to high-dimensional text.",
        "Linear SVM (LinearSVC) - maximum-margin linear classifier, very effective on "
        "high-dimensional sparse text features.",
        "Random Forest - non-linear ensemble; included to contrast tree-based learning "
        "against the linear models on sparse text.",
        "Multinomial Naive Bayes - probabilistic word-count model; classic, fast text "
        "classification baseline.",
    ]
    for i, p in enumerate(proposals, 1):
        print(f"  {i}. {p}")

    print(f"\nAll EDA charts saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
