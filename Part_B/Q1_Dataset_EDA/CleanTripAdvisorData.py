import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords


PART_B_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PART_B_DIR / "data" / "tripadvisor_raw.csv"
CLEAN_DATA_FILE = PART_B_DIR / "data" / "tripadvisor_clean.csv"

NEGATION_WORDS = {
    "no",
    "nor",
    "not",
    "never",
    "neither",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "without",
}
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS


def clean_review(text):
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Expand common contractions before punctuation removal so negation is retained.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def main():
    dataframe = pd.read_csv(RAW_DATA_FILE, usecols=["text", "label"])

    print("=== TripAdvisor Data Before Cleaning ===")
    print(f"Rows: {len(dataframe)}")
    print("Missing values:")
    print(dataframe[["text", "label"]].isnull().sum())
    print(f"Duplicate reviews: {dataframe.duplicated(subset=['text']).sum()}")
    print("Class distribution:")
    print(dataframe["label"].value_counts())

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe = dataframe.drop_duplicates(subset=["text"]).copy()
    dataframe["clean_text"] = dataframe["text"].map(clean_review)
    dataframe = dataframe[dataframe["clean_text"].str.strip() != ""].copy()

    dataframe["character_count"] = dataframe["text"].str.len()
    dataframe["word_count_before_cleaning"] = dataframe["text"].str.split().str.len()
    dataframe["word_count_after_cleaning"] = (
        dataframe["clean_text"].str.split().str.len()
    )

    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(CLEAN_DATA_FILE, index=False)

    print("\n=== TripAdvisor Data After Cleaning ===")
    print(f"Rows: {len(dataframe)}")
    print("Class distribution:")
    print(dataframe["label"].value_counts())
    not_count = dataframe["clean_text"].str.count(r"\bnot\b").sum()
    print(f"Negation 'not' occurrences: {not_count}")
    print(f"Saved cleaned dataset to: {CLEAN_DATA_FILE}")


if __name__ == "__main__":
    main()
