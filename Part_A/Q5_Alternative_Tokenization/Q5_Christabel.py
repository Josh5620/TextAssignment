from pathlib import Path

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Data_1.txt"


# Read the same text corpus used in Q1 so the outputs can be compared fairly.
with open(DATA_FILE, "r", encoding="utf-8") as file:
    text = file.read()


# Q1 used NLTK word_tokenize, so this is kept as the baseline method.
q1_tokens = word_tokenize(text)

# CountVectorizer is used as the alternative tokenization approach for Q5.
raw_vectorizer = CountVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
)
count_vectorizer_tokens = raw_vectorizer.build_analyzer()(text)

# CountVectorizer remove stop words automatically.
filtered_vectorizer = CountVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",
)
filtered_count_vectorizer_tokens = filtered_vectorizer.build_analyzer()(text)


# Print the original Q1 approach and token count for comparison.
print("=== Q1 Approach: NLTK word_tokenize ===")
print(q1_tokens)
print(f"Token count: {len(q1_tokens)}")

# Print the raw alternative tokenization output.
print("\n=== Q5 Alternative: CountVectorizer Analyzer ===")
print(count_vectorizer_tokens)
print(f"Token count: {len(count_vectorizer_tokens)}")

# Print the cleaned CountVectorizer output after stop-word and punctuation removal.
print("\n=== Q5 Alternative After Stop Word and Punctuation Removal ===")
print(filtered_count_vectorizer_tokens)
print(f"Filtered token count: {len(filtered_count_vectorizer_tokens)}")




