import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Load Clean Data ---
df = pd.read_csv('cyberbullying_clean.csv')

# --- Split into train/test (same split as Q2 so results stay comparable) ---
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['cyberbullying_type'],
    test_size=0.2, random_state=42, stratify=df['cyberbullying_type']
)

# --- Base pipeline to tune (same as Q2) ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_jobs=-1, random_state=42))
])

# --- Hyperparameter search space ---
param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__criterion': ['gini', 'entropy']
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
    name = param.replace('clf__', '')   
    print(f"  {name:<20} : {value}")
print("=" * 45)
print("Best CV Macro F1:", search.best_score_)

# --- Save tuned model for Q4 ---
joblib.dump(search.best_estimator_, 'rf_tuned.joblib')