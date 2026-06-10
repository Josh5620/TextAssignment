import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# Load the cleaned dataset created in Q1.
df = pd.read_csv(DATA_FILE)

# Use the same train-test split settings as Q2 so model results are comparable.
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["cyberbullying_type"],
    test_size=0.2,
    random_state=42,
    stratify=df["cyberbullying_type"],
)

# Create the base Linear SVM pipeline before tuning.
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC(max_iter=5000, random_state=42)),
    ]
)

# Hyperparameter search space for both TF-IDF feature extraction and Linear SVM.
param_dist = {
    "tfidf__max_features": [10000, 20000, 30000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
    "clf__class_weight": [None, "balanced"],
}

# RandomizedSearchCV tests several hyperparameter combinations using cross-validation.
# Macro F1 is used because it treats all classes equally, which is helpful for imbalanced data.
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=10,
    cv=5,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

search.fit(X_train, y_train)

# Report the best hyperparameters found for the Q3 discussion.
print("=== Q3: Linear SVM Best Hyperparameters ===")
print("=" * 45)
for param, value in search.best_params_.items():
    name = param.replace("clf__", "").replace("tfidf__", "")
    print(f"  {name:<20} : {value}")
print("=" * 45)
print("Best CV Macro F1:", search.best_score_)

# Save the tuned Linear SVM model for Q4 evaluation.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(search.best_estimator_, MODEL_DIR / "svm_tuned.joblib")
