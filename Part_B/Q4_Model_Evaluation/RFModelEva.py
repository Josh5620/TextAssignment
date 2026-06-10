from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_FILE = PART_B_DIR / "models" / "rf_tuned.joblib"


# --- Reload the trained pipeline ---
pipeline = joblib.load(MODEL_FILE)
vec = pipeline.named_steps['tfidf']
feature_names = vec.get_feature_names_out()

df = pd.read_csv(DATA_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['cyberbullying_type'],
    test_size=0.2, random_state=42, stratify=df['cyberbullying_type']
)

X_vec = vec.transform(X_train)

print("=== Top words per class ===")
for cls in sorted(y_train.unique()):
    target = (y_train == cls).astype(int)
    scores, _ = chi2(X_vec, target)
    top = np.array(feature_names)[scores.argsort()[::-1][:10]]
    print(f"\n{cls}:")
    print(", ".join(top))
    
print("=== Q3: Tuned Random Forest Classification Report ===")
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))
