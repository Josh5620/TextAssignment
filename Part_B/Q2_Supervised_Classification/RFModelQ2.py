import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_review_full_clean.csv"
MODEL_DIR = PART_B_DIR / "models"

# --- Load cleaned Yelp reviews ---
df = pd.read_csv(DATA_FILE)

# --- Split into train/test (stratify by 'rating' star label, random state for reproducibility) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['rating'],
    test_size=0.2, random_state=42, stratify=df['rating']
)
# --- Build RF pipeline ---
# tfidf: review text -> numeric features (10k vocab cap, single words + pairs)
# clf: 200 trees, parallel training, fixed randomness for the 5-class star rating
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
])

# --- Train, save for future use, predict based on test data ---
pipeline.fit(X_train, y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / 'rf_yelp_pipeline.joblib')
preds = pipeline.predict(X_test)

# --- Report base performance measures ---
print("=== Q2: Yelp Random Forest Classification Report ===")
print(classification_report(y_test, preds))
