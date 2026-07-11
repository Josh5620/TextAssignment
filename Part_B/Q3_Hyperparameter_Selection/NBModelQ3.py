import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_clean.csv"
MODEL_DIR = PART_B_DIR / "models"


# --- Load Clean Data (Yelp 3-class sentiment) ---
df = pd.read_csv(DATA_FILE)

# --- Split into train/test (same split as Q2 so results stay comparable) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'],
    test_size=0.2, random_state=42, stratify=df['sentiment']
)

# --- Base pipeline to tune (same as Q2) ---
# strip_accents is fixed; max_features/ngram_range/min_df/sublinear_tf are tuned below.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents="unicode")),
    ('clf', MultinomialNB())
])

# --- Hyperparameter search space ---
# Same TF-IDF search space across all 4 Q3 models for a fair comparison.
param_dist = {
    'tfidf__max_features': [10000, 20000, 30000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [1, 2, 3],
    'tfidf__sublinear_tf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0, 2.0],   # Laplace/Lidstone smoothing
    'clf__fit_prior': [True, False]       # learn class priors vs uniform
}

# --- Randomized search ---
# n_iter=10 tries 10 random combinations
# cv=5 uses 5-fold cross validation, scoring on macro F1 (equal weight per class)
search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=10, cv=5,
    scoring='f1_macro',
    random_state=42, n_jobs=-1, verbose=0
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
joblib.dump(search.best_estimator_, MODEL_DIR / 'nb_tuned.joblib')
