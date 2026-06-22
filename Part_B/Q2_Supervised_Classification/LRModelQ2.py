import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# --- Load Clean Data (Yelp 3-class sentiment) ---
df = pd.read_csv(DATA_FILE)

# --- Split into train/test (stratify by 'sentiment', random state for reproducibility) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'],
    test_size=0.2, random_state=42, stratify=df['sentiment']
)

# --- Build Logistic Regression pipeline ---
# tfidf: text -> numeric features. Shared config across all 4 Q2 models for a fair
#        comparison: 10k vocab, 1-2 grams, min_df=2, sublinear_tf, strip_accents.
# clf: multinomial logistic regression, balanced classes, more iters to converge
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                              min_df=2, sublinear_tf=True, strip_accents="unicode")),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                               n_jobs=-1, random_state=42))
])

# --- Train, save for future use, predict based on test data ---
pipeline.fit(X_train, y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / 'lr_pipeline.joblib')
preds = pipeline.predict(X_test)

# --- Report base performance measures ---
print("=== Q2: Logistic Regression Classification Report (Yelp 3-class sentiment) ===")
print(classification_report(y_test, preds))
