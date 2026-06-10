from collections import Counter
from fractions import Fraction
from pathlib import Path


START_TOKEN = "<s>"
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Data_3.txt"


def load_sentences(file_path):
    # Extract only the lines that are actual padded sentences.
    with open(file_path, "r", encoding="utf-8") as file:
        sentence_lines = [
            line.strip()
            for line in file
            if line.strip().startswith(START_TOKEN)
        ]

    # The first three padded sentences are the training corpus.
    training_sentences = [line.split() for line in sentence_lines[:-1]]

    # The final padded sentence is the sentence whose probability is calculated.
    test_sentence = sentence_lines[-1].split()
    return training_sentences, test_sentence


def get_bigrams(sentence):
    # Convert a sentence into adjacent word pairs such as (<s>, I), (I, read).
    return list(zip(sentence, sentence[1:]))


def format_fraction(value):
    # Keep exact fraction output for manual probability reporting.
    return f"{value.numerator}/{value.denominator}"


training_sentences, test_sentence = load_sentences(DATA_FILE)

# Count individual words and word pairs from the training corpus.
unigram_counts = Counter()
bigram_counts = Counter()

for sentence in training_sentences:
    unigram_counts.update(sentence)
    bigram_counts.update(get_bigrams(sentence))

# <s> is not included in the predicted vocabulary for add-one smoothing.
vocabulary = sorted(token for token in unigram_counts if token != START_TOKEN)
vocabulary_size = len(vocabulary)
test_bigrams = get_bigrams(test_sentence)


print("=== Training Sentences ===")
for sentence in training_sentences:
    print(" ".join(sentence))

print("\n=== Test Sentence ===")
print(" ".join(test_sentence))

print("\n=== Vocabulary ===")
print(vocabulary)
print(f"Vocabulary size excluding <s>: {vocabulary_size}")


print("\n=== Unsmoothed Bigram Model ===")
unsmoothed_probability = Fraction(1, 1)

# Unsmoothed formula:
# P(wi | wi-1) = Count(wi-1, wi) / Count(wi-1)
for previous_token, current_token in test_bigrams:
    bigram_count = bigram_counts[(previous_token, current_token)]
    previous_count = unigram_counts[previous_token]
    conditional_probability = Fraction(bigram_count, previous_count)
    unsmoothed_probability *= conditional_probability

    print(
        f"P({current_token} | {previous_token}) = "
        f"{bigram_count}/{previous_count} = {conditional_probability}"
    )

print(
    "Unsmoothed sentence probability = "
    f"{format_fraction(unsmoothed_probability)} = {float(unsmoothed_probability):.8f}"
)


print("\n=== Smoothed Bigram Model (Add-One / Laplace) ===")
smoothed_probability = Fraction(1, 1)

# Add-one smoothing formula:
# P(wi | wi-1) = (Count(wi-1, wi) + 1) / (Count(wi-1) + V)
for previous_token, current_token in test_bigrams:
    bigram_count = bigram_counts[(previous_token, current_token)]
    previous_count = unigram_counts[previous_token]
    numerator = bigram_count + 1
    denominator = previous_count + vocabulary_size
    conditional_probability = Fraction(numerator, denominator)
    smoothed_probability *= conditional_probability

    print(
        f"P({current_token} | {previous_token}) = "
        f"({bigram_count}+1)/({previous_count}+{vocabulary_size}) = "
        f"{numerator}/{denominator} = {conditional_probability}"
    )

print(
    "Smoothed sentence probability = "
    f"{format_fraction(smoothed_probability)} = {float(smoothed_probability):.8f}"
)
