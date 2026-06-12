"""
Dataset bake-off: run the SAME pipeline on several candidate datasets and
compare them against the Part B marking criteria.

Pipeline per dataset:  raw text -> our cleaning -> stratified subsample (cap 20k)
-> TF-IDF (1-2 gram, 10k) -> Logistic Regression vs Multinomial NB.

We report: rows used, #classes, majority-class %, raw median words (length),
and LR/NB accuracy + macro-F1, plus the macro-F1 spread between the two models.
A good dataset for this assignment = medium size, longer/raw text, and a
non-trivial gap (real model spread) without a trivial accuracy ceiling.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATA = Path(__file__).resolve().parents[1] / "data"
DOWNLOADS = Path.home() / "Downloads"
STOP = set(ENGLISH_STOP_WORDS)
CAP = 20000
JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags (IMDb has <br />)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOP and len(t) > 1)


# ---- loaders: each returns a DataFrame with columns text, label ----
def load_jigsaw():
    df = pd.read_csv(DATA / "jigsaw_train.csv")
    df["label"] = (df[JIGSAW_LABELS].sum(axis=1) > 0).map({True: "toxic", False: "clean"})
    return df.rename(columns={"comment_text": "text"})[["text", "label"]]


def load_fpb():
    df = pd.read_csv(DOWNLOADS / "data.csv")
    return df.rename(columns={"Sentence": "text", "Sentiment": "label"})[["text", "label"]]


def load_imdb():
    df = pd.read_parquet(DATA / "imdb_train.parquet")
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    return df[["text", "label"]]


def load_enron():
    df = pd.read_csv(DATA / "enron_spam.csv")
    df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
    return df.rename(columns={"Spam/Ham": "label"})[["text", "label"]]


def load_ott():
    z = zipfile.ZipFile(DATA / "op_spam_v1.4.zip")
    rows = []
    for n in z.namelist():
        if n.endswith(".txt"):
            label = "deceptive" if "deceptive_from_MTurk" in n else "truthful"
            rows.append({"text": z.read(n).decode("utf-8", "ignore"), "label": label})
    return pd.DataFrame(rows)


def _rating_to_sentiment(df, rating_col, text_col):
    # 1-2 stars -> negative, 4-5 -> positive, drop the neutral 3-star reviews
    out = df.rename(columns={text_col: "text"})[["text"]].copy()
    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    out = out[out["rating"].isin([1, 2, 4, 5])]
    out["label"] = (out["rating"] >= 4).map({True: "positive", False: "negative"})
    return out[["text", "label"]]


def load_fakereviews():
    df = pd.read_csv(DATA / "fake reviews dataset.csv")
    df["label"] = df["label"].map({"CG": "fake", "OR": "real"}).fillna(df["label"])
    return df.rename(columns={"text_": "text"})[["text", "label"]]


def load_tripadvisor():
    df = pd.read_csv(DATA / "tripadvisor_hotel_reviews.csv")
    return _rating_to_sentiment(df, "Rating", "Review")


def load_womens():
    df = pd.read_csv(DATA / "Womens Clothing E-Commerce Reviews.csv")
    return _rating_to_sentiment(df, "Rating", "Review Text")


DATASETS = [
    ("Fake Reviews 40k (fake review)", load_fakereviews),
    ("TripAdvisor 20k (sentiment)", load_tripadvisor),
    ("Womens Clothing (sentiment)", load_womens),
    ("Jigsaw (cyberbully)", load_jigsaw),
    ("FinancialPhraseBank (sentiment)", load_fpb),
    ("IMDb (sentiment/movie)", load_imdb),
    ("Enron-Spam (spam)", load_enron),
    ("OpSpam/Ott (fake review)", load_ott),
]


def evaluate(df):
    df = df.dropna(subset=["text", "label"]).copy()
    raw_med = int(df["text"].astype(str).str.split().str.len().median())
    n_classes = df["label"].nunique()
    majority = round(df["label"].value_counts(normalize=True).max() * 100, 1)

    if len(df) > CAP:
        df, _ = train_test_split(df, train_size=CAP, random_state=42, stratify=df["label"])

    df["clean"] = df["text"].apply(clean)
    df = df[df["clean"].str.strip() != ""]

    Xtr, Xte, ytr, yte = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    Xtr_v, Xte_v = vec.fit_transform(Xtr), vec.transform(Xte)

    out = {"rows": len(df), "classes": n_classes, "majority%": majority, "raw_med_words": raw_med}
    f1s = {}
    for name, model in [("LR", LogisticRegression(max_iter=2000, class_weight="balanced")),
                        ("NB", MultinomialNB())]:
        model.fit(Xtr_v, ytr)
        pred = model.predict(Xte_v)
        out[f"{name}_acc"] = round(accuracy_score(yte, pred), 3)
        f1 = f1_score(yte, pred, average="macro")
        out[f"{name}_mF1"] = round(f1, 3)
        f1s[name] = f1
    out["mF1_spread"] = round(abs(f1s["LR"] - f1s["NB"]), 3)
    return out


def main():
    results = []
    for name, loader in DATASETS:
        print(f"Running {name} ...")
        try:
            res = evaluate(loader())
            res["dataset"] = name
            results.append(res)
        except Exception as e:
            print(f"  FAILED: {e}")

    cols = ["dataset", "rows", "classes", "majority%", "raw_med_words",
            "LR_acc", "LR_mF1", "NB_acc", "NB_mF1", "mF1_spread"]
    table = pd.DataFrame(results)[cols]
    print("\n================ BAKE-OFF RESULTS ================")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
