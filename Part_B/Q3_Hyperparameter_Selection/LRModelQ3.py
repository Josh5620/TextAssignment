import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# --- Load Clean Data ---
df = pd.read_csv(DATA_FILE)

# --- Split into train/test (same split as Q2 so results stay comparable) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['cyberbullying_type'],
    test_size=0.2, random_state=42, stratify=df['cyberbullying_type']
)

# --- Base pipeline to tune (same as Q2) ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42))
])

# --- Hyperparameter search space ---
param_dist = {
    'tfidf__max_features': [10000, 20000, 30000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'clf__penalty': ['l2'],
    'clf__class_weight': [None, 'balanced']
}

# --- Randomized search ---
# n_iter=10 tries 10 random combinations
# cv=5 uses 5-fold cross validation, scoring on macro F1 (balanced classes)
search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=10, cv=5,
    scoring='f1_macro',
    random_state=42, n_jobs=-1, verbose=2
)

search.fit(X_train, y_train)

# --- Report best hyperparameters and score ---
print("=== Q3: Best Hyperparameters ===")
print("=" * 45)
for param, value in search.best_params_.items():
    name = param.replace('clf__', '').replace('tfidf__', '')
    print(f"  {name:<20} : {value}")
print("=" * 45)
print("Best CV Macro F1:", search.best_score_)

# --- Save tuned model for Q4 ---
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(search.best_estimator_, MODEL_DIR / 'lr_tuned.joblib')
