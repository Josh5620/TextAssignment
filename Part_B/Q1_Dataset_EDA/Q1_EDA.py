from pathlib import Path
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PART_B_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "cyberbullying_tweets.csv"
CLEAN_DATA_FILE = DATA_DIR / "cyberbullying_clean.csv"
TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "cyberbullying_type"


# Prepare reusable NLP tools for tweet cleaning.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_tweet(text):
    # Convert each tweet into a cleaner text form for text classification.
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs.
    text = re.sub(r"@\w+", "", text)  # Remove user mentions.
    text = re.sub(r"#(\w+)", r"\1", text)  # Keep hashtag word but remove #.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Reduce repeated letters.
    text = re.sub(r"[^a-z\s]", "", text)  # Keep only letters and spaces.

    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]  # Remove common words.
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Reduce words to base form.
    return " ".join(tokens)


def print_section(title):
    print(f"\n=== {title} ===")


def print_top_words(texts, title, top_n=20):
    # CountVectorizer is used here only for EDA word-frequency analysis.
    vectorizer = CountVectorizer(max_features=top_n)
    word_matrix = vectorizer.fit_transform(texts)
    word_counts = word_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    ranked_words = sorted(zip(words, word_counts), key=lambda item: item[1], reverse=True)

    print_section(title)
    for word, count in ranked_words:
        print(f"{word:<20} {count}")


def main():
    raw_file_exists = RAW_DATA_FILE.exists()
    clean_file_exists = CLEAN_DATA_FILE.exists()

    if raw_file_exists:
        data_file = RAW_DATA_FILE
        print_section("Data Source")
        print(f"Using raw dataset: {RAW_DATA_FILE.name}")
    elif clean_file_exists:
        data_file = CLEAN_DATA_FILE
        print_section("Data Source")
        print(f"Using existing cleaned dataset: {CLEAN_DATA_FILE.name}")
    else:
        raise FileNotFoundError(
            f"Place either {RAW_DATA_FILE.name} or {CLEAN_DATA_FILE.name} in {DATA_DIR} before running Q1_EDA.py."
        )

    # Load the selected secondary dataset.
    df = pd.read_csv(data_file)

    # Basic dataset description for Q1.1.
    print_section("Dataset Overview")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {list(df.columns)}")

    missing_columns = [column for column in [TEXT_COLUMN, LABEL_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Show example rows so the report can describe the input and target label.
    print_section("Sample Records")
    print(df[[TEXT_COLUMN, LABEL_COLUMN]].head())

    # Check data quality issues before cleaning.
    print_section("Missing Values")
    print(df[[TEXT_COLUMN, LABEL_COLUMN]].isnull().sum())

    print_section("Duplicate Tweets")
    print(f"Duplicate rows: {df.duplicated(subset=[TEXT_COLUMN]).sum()}")

    print_section("Class Distribution")
    print(df[LABEL_COLUMN].value_counts())

    # Apply text preprocessing and remove unusable empty rows.
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()

    if "clean" not in df.columns:
        df["clean"] = df[TEXT_COLUMN].apply(clean_tweet)
    else:
        df["clean"] = df["clean"].fillna("").astype(str)

    df = df[df["clean"].str.strip() != ""].copy()

    # Create simple text-length features for EDA comparison.
    df["character_count"] = df[TEXT_COLUMN].astype(str).str.len()
    df["word_count_before_cleaning"] = df[TEXT_COLUMN].astype(str).str.split().str.len()
    df["word_count_after_cleaning"] = df["clean"].str.split().str.len()

    print_section("Text Length Summary")
    print(
        df[
            [
                "character_count",
                "word_count_before_cleaning",
                "word_count_after_cleaning",
            ]
        ].describe()
    )

    print_section("Average Cleaned Word Count By Class")
    print(df.groupby(LABEL_COLUMN)["word_count_after_cleaning"].mean().sort_values(ascending=False))

    print_top_words(df["clean"], "Top Words After Cleaning")

    # Suggested models are based on common supervised text-classification methods.
    print_section("Suggested Predictive Models")
    suggested_models = [
        "Logistic Regression: strong baseline for TF-IDF text classification.",
        "Linear SVM: effective for high-dimensional sparse text features.",
        "Random Forest: non-linear model that can capture feature interactions.",
    ]
    for model in suggested_models:
        print(f"- {model}")

    # Save the prepared dataset for Q2, Q3, and Q4 modelling scripts.
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_FILE, index=False)

    print_section("Saved Clean Dataset")
    print(f"Saved cleaned dataset to {CLEAN_DATA_FILE}")
    print(f"Cleaned rows: {len(df)}")


if __name__ == "__main__":
    main()
