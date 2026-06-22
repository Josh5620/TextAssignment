import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_clean.csv"
MODEL_FILE = PART_B_DIR / "models" / "svm_tuned.joblib"


# Load the tuned Yelp Linear SVM pipeline saved by Q3.
pipeline = joblib.load(MODEL_FILE)
vectorizer = pipeline.named_steps["tfidf"]
classifier = pipeline.named_steps["clf"]
feature_names = vectorizer.get_feature_names_out()

# Recreate the same held-out test set used throughout the SVM workflow.
df = pd.read_csv(DATA_FILE, usecols=["clean_text", "sentiment"])
df = df.dropna(subset=["clean_text", "sentiment"]).copy()
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)

# Use the signed Linear SVM coefficients to identify class-specific terms.
print("=== Top Terms Per Sentiment Class ===")
coefficients = classifier.coef_
classes = classifier.classes_

if len(classes) == 2 and coefficients.shape[0] == 1:
    negative_terms = np.array(feature_names)[coefficients[0].argsort()[:10]]
    positive_terms = np.array(feature_names)[coefficients[0].argsort()[::-1][:10]]

    print(f"\n{classes[0]}:")
    print(", ".join(negative_terms))
    print(f"\n{classes[1]}:")
    print(", ".join(positive_terms))
else:
    for class_index, class_name in enumerate(classes):
        top_terms = np.array(feature_names)[
            coefficients[class_index].argsort()[::-1][:10]
        ]
        print(f"\n{class_name}:")
        print(", ".join(top_terms))

# Evaluate the tuned model only once on the unseen test set.
predictions = pipeline.predict(X_test)
labels = sorted(y_test.unique())
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)
weighted_f1 = precision_recall_fscore_support(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)[2]

print("\n=== Q4: Tuned Yelp Linear SVM 3-Class Classification Report ===")
print(classification_report(y_test, predictions, digits=4, zero_division=0))

print("=== Q4: Tuned Yelp Linear SVM Summary ===")
print(f"Accuracy:        {accuracy_score(y_test, predictions):.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1-score:  {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("\n=== Q4: Tuned Yelp Linear SVM Confusion Matrix ===")
print(confusion_matrix(y_test, predictions, labels=labels))
print("Class order:", labels)
