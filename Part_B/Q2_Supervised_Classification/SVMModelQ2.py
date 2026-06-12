import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "tripadvisor_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# Load the negation-aware cleaned TripAdvisor reviews and sentiment labels.
df = pd.read_csv(DATA_FILE, usecols=["clean_text", "label"])
df = df.dropna(subset=["clean_text", "label"]).copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]

# Use a reproducible stratified 80:20 split to preserve the class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# The cleaned text retains negation; bigrams capture phrases such as "not good".
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

# Train and save the baseline TripAdvisor Linear SVM model.
pipeline.fit(X_train, y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / "svm_tripadvisor_pipeline.joblib")

# Evaluate the baseline model on the unseen test set.
predictions = pipeline.predict(X_test)

print("=== Q2: TripAdvisor Linear SVM Classification Report ===")
print(f"Rows used: {len(df)}")
print("Class distribution:")
print(df["label"].value_counts())
print()
print(classification_report(y_test, predictions, digits=4, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
