"""
Throwaway benchmark: does the new STANDARD clean_review() cost us accuracy vs the
feature-engineered clean_review_v2 (which has neg_ marking + emoticon tokens)?

Compares three cleanings on the SAME rows + SAME split (3-class Yelp sentiment):
  A. clean_review_v2            -> feature-engineered (neg_ + emoticons)   [current]
  B. clean_review(keep_neg=True)-> standard cleaning, "not" kept as a token
  C. clean_review(keep_neg=False)-> pure textbook, negation words dropped too

Isolates two costs: A vs B = the neg_/emoticon feature-engineering gain;
B vs C = keeping negation words. Helps the "re-add neg_ later?" decision.
Diagnostic only -- writes no csv, touches nothing.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from clean_review_v2 import clean_review_v2
from clean_review import clean_review

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "yelp_review_full_raw_30k.csv"

N_SAMPLE = 20000   # stratified-ish subsample so the 3 cleaning passes stay quick

df = pd.read_csv(RAW).dropna(subset=["review", "rating"])
if len(df) > N_SAMPLE:
    df = df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
df["sentiment"] = np.where(df["rating"] <= 2, "negative",
                           np.where(df["rating"] == 3, "neutral", "positive"))

print(f"Cleaning {len(df)} reviews three ways (this takes a few minutes for POS tagging)...")
df["A_v2"] = df["review"].apply(clean_review_v2)
df["B_keepneg"] = df["review"].apply(lambda t: clean_review(t, keep_negation=True))
df["C_textbook"] = df["review"].apply(lambda t: clean_review(t, keep_negation=False))

# keep only rows where ALL three variants are non-empty, so every variant is
# evaluated on the identical row set + identical split
for col in ["A_v2", "B_keepneg", "C_textbook"]:
    df[col] = df[col].str.strip()
df = df[(df["A_v2"] != "") & (df["B_keepneg"] != "") & (df["C_textbook"] != "")].reset_index(drop=True)
print(f"Rows used (non-empty in all variants): {len(df)}\n")

# show a few before/after examples
print("=" * 70)
print("SAMPLE before/after (first 3 reviews)")
print("=" * 70)
for i in range(3):
    print(f"\nRAW : {df['review'].iloc[i][:140]}")
    print(f"  A v2        : {df['A_v2'].iloc[i][:120]}")
    print(f"  B keep_neg  : {df['B_keepneg'].iloc[i][:120]}")
    print(f"  C textbook  : {df['C_textbook'].iloc[i][:120]}")

# one shared stratified split (indices), reused for all three variants
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=df["sentiment"])
y_tr, y_te = df["sentiment"].iloc[tr], df["sentiment"].iloc[te]

models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVM": LinearSVC(class_weight="balanced"),
    "NaiveBayes": MultinomialNB(),
}
variants = {
    "A v2 (neg_+emoji)": "A_v2",
    "B clean keep_neg ": "B_keepneg",
    "C clean textbook ": "C_textbook",
}

print("\n" + "=" * 70)
print("MACRO-F1 by cleaning variant (same rows, same split)")
print("=" * 70)
header = "variant            | " + " | ".join(f"{m:>10}" for m in models)
print(header)
print("-" * len(header))
for vname, col in variants.items():
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(df[col].iloc[tr])
    Xte = vec.transform(df[col].iloc[te])
    cells = []
    for clf in models.values():
        clf.fit(Xtr, y_tr)
        f1 = f1_score(y_te, clf.predict(Xte), average="macro")
        cells.append(f"{f1:>10.3f}")
    print(f"{vname} | " + " | ".join(cells))

print("\nReading: A vs B = cost of dropping neg_/emoticon feature engineering;")
print("         B vs C = cost of dropping negation words entirely.")
