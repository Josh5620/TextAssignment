import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# Load the same cleaned three-class Yelp dataset used by the Q2 Linear SVM model.
df = pd.read_csv(DATA_FILE, usecols=["clean_text", "sentiment"])
df = df.dropna(subset=["clean_text", "sentiment"]).copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]

# Recreate the same split so Q2, Q3, and Q4 results remain comparable.
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)

# Negation is preserved by the cleaning script before TF-IDF is applied.
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LinearSVC(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

# Tune both TF-IDF feature extraction and Linear SVM hyperparameters.
param_dist = {
    "tfidf__max_features": [10000, 20000, 30000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}

# Macro F1 gives equal importance to negative, neutral, and positive classes.
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=20,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

search.fit(X_train, y_train)

print("=== Q3: Yelp Linear SVM Best Hyperparameters (3-class sentiment) ===")
print("=" * 55)
for param, value in search.best_params_.items():
    name = param.replace("clf__", "").replace("tfidf__", "")
    print(f"  {name:<20} : {value}")
print("=" * 55)
print(f"Best CV Macro F1: {search.best_score_:.4f}")

# Save the tuned Yelp Linear SVM model for Q4.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    search.best_estimator_,
    MODEL_DIR / "svm_tuned.joblib",
)
