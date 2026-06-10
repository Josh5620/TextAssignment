import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_FILE = PART_B_DIR / "models" / "svm_tuned.joblib"


# Load the tuned Linear SVM pipeline saved from Q3.
pipeline = joblib.load(MODEL_FILE)
vectorizer = pipeline.named_steps["tfidf"]
feature_names = vectorizer.get_feature_names_out()

# Load the cleaned dataset and recreate the same train-test split used earlier.
df = pd.read_csv(DATA_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["cyberbullying_type"],
    test_size=0.2,
    random_state=42,
    stratify=df["cyberbullying_type"],
)

# Transform training text into TF-IDF features for keyword analysis.
X_train_vectorized = vectorizer.transform(X_train)

print("=== Top Words Per Class ===")
for class_name in sorted(y_train.unique()):
    # Chi-square scores identify words that are strongly associated with each class.
    target_class = (y_train == class_name).astype(int)
    scores, _ = chi2(X_train_vectorized, target_class)
    top_words = np.array(feature_names)[scores.argsort()[::-1][:10]]

    print(f"\n{class_name}:")
    print(", ".join(top_words))

# Evaluate the tuned Linear SVM model on the unseen test set.
predictions = pipeline.predict(X_test)

print("\n=== Q4: Tuned Linear SVM Classification Report ===")
print(classification_report(y_test, predictions))

print("=== Q4: Tuned Linear SVM Confusion Matrix ===")
print(confusion_matrix(y_test, predictions, labels=sorted(y_test.unique())))
print("Class order:", sorted(y_test.unique()))
