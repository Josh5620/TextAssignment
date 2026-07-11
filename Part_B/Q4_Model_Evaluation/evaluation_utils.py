import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


EXPECTED_LABELS = ["negative", "neutral", "positive"]


def load_yelp_split(data_file):
    """Load the cleaned Yelp data and recreate the shared 80:20 test split."""
    df = pd.read_csv(data_file, usecols=["clean_text", "sentiment"])
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
    return df, X_train, X_test, y_train, y_test


def validate_pipeline_labels(pipeline, model_name):
    """Stop stale/wrong saved models from being evaluated on the Yelp labels."""
    classifier = pipeline.named_steps["clf"]
    classes = list(getattr(classifier, "classes_", []))
    if sorted(classes) != EXPECTED_LABELS:
        raise ValueError(
            f"{model_name} is trained on {classes}, but this Part B task expects "
            f"{EXPECTED_LABELS}. Re-run the matching Q3 script to regenerate the "
            "Yelp tuned model before Q4 evaluation."
        )


def print_top_terms(pipeline, X_train, y_train, model_name, top_n=10):
    """Print interpretable class-associated terms for the report discussion."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(classifier.classes_)

    print(f"=== Top Terms Per Sentiment Class ({model_name}) ===")

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        if len(classes) == 2 and coefficients.shape[0] == 1:
            negative_terms = feature_names[coefficients[0].argsort()[:top_n]]
            positive_terms = feature_names[coefficients[0].argsort()[::-1][:top_n]]
            print(f"\n{classes[0]}:")
            print(", ".join(negative_terms))
            print(f"\n{classes[1]}:")
            print(", ".join(positive_terms))
        else:
            for class_index, class_name in enumerate(classes):
                top_terms = feature_names[
                    coefficients[class_index].argsort()[::-1][:top_n]
                ]
                print(f"\n{class_name}:")
                print(", ".join(top_terms))
        return

    if hasattr(classifier, "feature_log_prob_"):
        log_probs = classifier.feature_log_prob_
        for class_index, class_name in enumerate(classes):
            other_log_probs = np.delete(log_probs, class_index, axis=0).mean(axis=0)
            class_lift = log_probs[class_index] - other_log_probs
            top_terms = feature_names[class_lift.argsort()[::-1][:top_n]]
            print(f"\n{class_name}:")
            print(", ".join(top_terms))
        return

    X_vec = vectorizer.transform(X_train)
    for class_name in EXPECTED_LABELS:
        mask = (y_train == class_name).values
        target = mask.astype(int)
        scores, _ = chi2(X_vec, target)

        # Keep terms that are more common inside this class, so each list is
        # class-specific instead of repeating terms that are merely absent.
        mean_in = np.asarray(X_vec[mask].mean(axis=0)).ravel()
        mean_out = np.asarray(X_vec[~mask].mean(axis=0)).ravel()
        characteristic = mean_in > mean_out

        order = np.argsort(scores)[::-1]
        picks = [i for i in order if characteristic[i]][:top_n]
        print(f"\n{class_name}:")
        print(", ".join(feature_names[picks]))


def print_evaluation(pipeline, X_test, y_test, model_name):
    """Print the Q4 report metrics in a consistent format for all models."""
    predictions = pipeline.predict(X_test)
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

    print(f"\n=== Q4: Tuned Yelp {model_name} 3-Class Classification Report ===")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    print(f"=== Q4: Tuned Yelp {model_name} Summary ===")
    print(f"Accuracy:        {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print(f"\n=== Q4: Tuned Yelp {model_name} Confusion Matrix ===")
    print(confusion_matrix(y_test, predictions, labels=EXPECTED_LABELS))
    print("Class order:", EXPECTED_LABELS)
