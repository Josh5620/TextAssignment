"""
clean_review — a STANDARD text-cleaning / preprocessing function for the Part B
sentiment pipeline.

This is the "cleaning" stage proper: it only normalises and removes noise. It does
NOT do feature engineering (no `neg_` negation-marking, no emoticon->sentiment
tokens) — those inject sentiment signal and belong in a separate Feature
Engineering stage, not here. (See clean_review_v2.py for the feature-engineered
version, and Pang, Lee & Vaithyanathan 2002 for the `neg_` technique.)

The steps follow the standard text-preprocessing pipeline described in common
references:
  - Analytics Vidhya, "Text Cleaning Methods in NLP" (lowercase, remove
    punctuation/numbers, remove emojis/emoticons, expand contractions)
  - Towards Data Science, "Cleaning & Preprocessing Text Data for Sentiment
    Analysis" (lowercase, remove punctuation, remove emojis, remove stopwords,
    lemmatize)
  - GeeksforGeeks, "What is Sentiment Analysis?" (clean -> tokenize -> stopword
    removal -> lemmatization, with feature representation kept as a SEPARATE stage)

Steps:
  1. lowercase + normalise the curly apostrophe
  2. strip leftover literal escape codes (\\n \\t \\r) seen in scraped reviews
  3. remove URLs and HTML tags
  4. expand contractions (so "don't" -> "do not"); keeps the word "not" alive
  5. remove text emoticons (textbook default is to DELETE them, not encode them)
  6. tokenise letters-only -> drops punctuation and numbers in one step
  7. remove English stopwords + single-character tokens
  8. POS-aware lemmatization (loved -> love, better -> good)

`keep_negation` (default True): negation words ("no", "not", "never", ...) are
normally English stopwords and would be deleted in step 7. Keeping them is a
small, well-justified refinement for sentiment (so "not bad" survives as the two
tokens "not" and "bad"). Set keep_negation=False for a pure textbook clean that
drops them too.

Usage:
    from clean_review import clean_review
    df["clean_text"] = df["text"].apply(clean_review)
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Reproducibility guard: make sure the NLTK data this cleaner needs is present
# (so it runs top-to-bottom on a fresh machine). Each resource is tried
# independently so an id that doesn't exist on a given NLTK version cannot abort
# the whole run.
_REQUIRED_NLTK = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}
for _path, _pkg in _REQUIRED_NLTK.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass  # not fatal: a missing optional id (e.g. version-renamed) is tolerated

# Negation words. Standard cleaning would delete these (they are stopwords); the
# keep_negation flag below decides whether to protect them.
NEGATION_WORDS = {
    "no", "nor", "not", "never", "neither", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without",
}
_FULL_STOPWORDS = set(stopwords.words("english"))
# Two ready stoplists, picked per-call by the keep_negation flag.
_STOP_KEEP_NEG = _FULL_STOPWORDS - NEGATION_WORDS   # keep "not" etc.
_STOP_TEXTBOOK = _FULL_STOPWORDS                    # drop everything (pure textbook)

_lemmatizer = WordNetLemmatizer()
_lemma_cache = {}

# Emoticons to DELETE (cleaning, not feature engineering). We remove them
# explicitly before the letters-only tokenizer so faces like ":D" / ":P" don't
# leave a stray "d"/"p" token behind. Patterns are lowercase (text is lowercased
# first).
_EMOTICONS = re.compile(
    r"(:-?\)|:-?\]|=\)|\(:|:-?d|=d|;-?\)|:'\)|:-?\(|:-?\[|=\(|\):|:-?/|:-?\\|:'\(|d:)"
)


def _penn_to_wordnet(tag):
    # Map a Penn Treebank POS tag to the WordNet POS the lemmatizer expects.
    if tag.startswith("J"):
        return "a"   # adjective
    if tag.startswith("V"):
        return "v"   # verb
    if tag.startswith("R"):
        return "r"   # adverb
    return "n"       # default: noun


def _lemmatize(word, pos):
    key = (word, pos)
    if key not in _lemma_cache:
        _lemma_cache[key] = _lemmatizer.lemmatize(word, pos)
    return _lemma_cache[key]


def _expand_contractions(text):
    # Expand contractions BEFORE punctuation is removed, so "not" survives.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def clean_review(text, keep_negation=True):
    # 1. lowercase and normalise the curly apostrophe
    text = str(text).lower().replace("’", "'")
    # 2. strip leftover literal escape codes (\n \t \r) seen in scraped text
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    # 3. remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # 4. expand negation-bearing contractions
    text = _expand_contractions(text)
    # 5. remove emoticons (delete, not encode -- standard cleaning)
    text = _EMOTICONS.sub(" ", text)

    # 6. tokenise letters-only -> drops punctuation and numbers
    tokens = re.findall(r"[a-z]+", text)

    # 7 + 8. drop stopwords / single letters, then POS-lemmatize the rest
    stop = _STOP_KEEP_NEG if keep_negation else _STOP_TEXTBOOK
    tags = nltk.pos_tag(tokens)
    cleaned = [
        _lemmatize(tok, _penn_to_wordnet(tag))
        for tok, tag in tags
        if tok not in stop and len(tok) > 1
    ]
    return " ".join(cleaned)


if __name__ == "__main__":
    samples = [
        "The food was not bad at all, but the service wasn't great.",
        "I do not like this place. It was good though.",
        "Absolutely loved it! Best meal ever, would definitely return. :)",
        "Cannot recommend this dump. Waited 30 mins for cold fries :(",
    ]
    for s in samples:
        print("RAW           :", s)
        print("CLEAN keep_neg:", clean_review(s, keep_negation=True))
        print("CLEAN textbook:", clean_review(s, keep_negation=False))
        print()
