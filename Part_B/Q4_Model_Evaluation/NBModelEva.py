from pathlib import Path
import joblib

from evaluation_utils import (
    load_yelp_split,
    print_evaluation,
    print_top_terms,
    validate_pipeline_labels,
)


PART_B_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PART_B_DIR / "data" / "yelp_clean.csv"
MODEL_FILE = PART_B_DIR / "models" / "nb_tuned.joblib"


# Reload the tuned Yelp Multinomial Naive Bayes pipeline saved by Q3.
pipeline = joblib.load(MODEL_FILE)
validate_pipeline_labels(pipeline, "Multinomial Naive Bayes")

# Recreate the same held-out test set used throughout Part B.
df, X_train, X_test, y_train, y_test = load_yelp_split(DATA_FILE)

print_top_terms(pipeline, X_train, y_train, "Multinomial Naive Bayes")
print_evaluation(pipeline, X_test, y_test, "Multinomial Naive Bayes")
