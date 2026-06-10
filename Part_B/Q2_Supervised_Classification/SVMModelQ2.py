import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# Load the cleaned dataset created during Q1 data preparation.
df = pd.read_csv(DATA_FILE)

# Split the cleaned tweet text and target labels into training and testing sets.
# Stratify keeps the cyberbullying classes in similar proportions across both sets.
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["cyberbullying_type"],
    test_size=0.2,
    random_state=42,
    stratify=df["cyberbullying_type"],
)

# Build a supervised text-classification pipeline.
# TfidfVectorizer converts text into numerical TF-IDF features.
# LinearSVC applies a Linear Support Vector Machine classifier to those features.
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000, random_state=42)),
    ]
)

# Train the Linear SVM model and save it for later evaluation.
pipeline.fit(X_train, y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / "svm_pipeline.joblib")

# Predict the cyberbullying class for the unseen test tweets.
predictions = pipeline.predict(X_test)

# Print precision, recall, F1-score, and accuracy for Q2 model reporting.
print("=== Q2: Linear SVM Classification Report ===")
print(classification_report(y_test, predictions))
