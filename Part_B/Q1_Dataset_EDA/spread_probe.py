"""
Quick probe: which class framing (5-class / 3-class / binary) gives the biggest
SPREAD among the 4 models (best - worst)? A bigger spread = the model choice
matters more = a more interesting 4-model comparison for the report.

Throwaway diagnostic, not part of the EDA deliverable.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "yelp_clean.csv"

df = pd.read_csv(DATA).dropna(subset=["clean_text", "rating"])
df = df[df["clean_text"].str.strip() != ""]

def to_3class(s):
    return np.where(s <= 2, "neg", np.where(s == 3, "neu", "pos"))

framings = {}
# 5-class: raw stars
framings["5-class (1..5)"] = (df["clean_text"], df["rating"].astype(str))
# 3-class: neg / neu / pos
framings["3-class (neg/neu/pos)"] = (df["clean_text"], pd.Series(to_3class(df["rating"]), index=df.index))
# binary: drop neutral 3-star, neg vs pos
bin_mask = df["rating"] != 3
framings["binary (neg vs pos)"] = (
    df.loc[bin_mask, "clean_text"],
    np.where(df.loc[bin_mask, "rating"] <= 2, "neg", "pos"),
)

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "LinearSVM": LinearSVC(),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "NaiveBayes": MultinomialNB(),
}

for fname, (X_text, y) in framings.items():
    y = pd.Series(y).reset_index(drop=True)
    X_text = pd.Series(X_text).reset_index(drop=True)
    Xtr_t, Xte_t, ytr, yte = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_t)
    Xte = vec.transform(Xte_t)

    print(f"\n=== {fname}  (n={len(y)}, classes={y.nunique()}) ===")
    accs, f1s = {}, {}
    for mname, clf in models.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        accs[mname], f1s[mname] = acc, f1
        print(f"  {mname:13s}  acc={acc:.3f}  macroF1={f1:.3f}")
    acc_spread = max(accs.values()) - min(accs.values())
    f1_spread = max(f1s.values()) - min(f1s.values())
    best = max(f1s, key=f1s.get)
    worst = min(f1s, key=f1s.get)
    print(f"  --> SPREAD  acc={acc_spread:.3f}  macroF1={f1_spread:.3f}"
          f"   (best={best}, worst={worst})")
