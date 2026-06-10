import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# --- Load Clean Data ---
df = pd.read_csv(DATA_FILE)

# --- Split into train/test (stratify by 'cyberbullying_type', random state for reproducibility) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['cyberbullying_type'],
    test_size=0.2, random_state=42, stratify=df['cyberbullying_type']
)

# --- Build Logistic Regression pipeline ---
# tfidf: text -> numeric features (10k vocab cap, single words + pairs)
# clf: multinomial logistic regression, balanced classes, more iters to converge
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                               n_jobs=-1, random_state=42))
])

# --- Train, save for future use, predict based on test data ---
pipeline.fit(X_train, y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / 'lr_pipeline.joblib')
preds = pipeline.predict(X_test)

# --- Report base performance measures ---
print("=== Q2: Logistic Regression Classification Report ===")
print(classification_report(y_test, preds))
