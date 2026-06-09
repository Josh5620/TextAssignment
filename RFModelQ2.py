import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Load Clean Data ---
df = pd.read_csv('cyberbullying_clean.csv')

# --- Split into train/test (stratify by 'cyberbullying_type', random state for reproducibility) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['cyberbullying_type'],
    test_size=0.2, random_state=42, stratify=df['cyberbullying_type']
)

# --- Build RF pipeline ---
# tfidf: text -> numeric features (10k vocab cap, single words + pairs)
# clf: 200 trees, parallel training, fixed randomness
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
])

# --- Train, save for future use, predict based on test data ---
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'rf_pipeline.joblib')
preds = pipeline.predict(X_test)

# --- Report base performance measures ---
print("=== Q2: Random Forest Classification Report ===")
print(classification_report(y_test, preds))