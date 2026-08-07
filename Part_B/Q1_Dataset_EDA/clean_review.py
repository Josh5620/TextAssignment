"""
Text cleaning for the Part B sentiment pipeline.

Turns a raw review into a space-separated string of lemmas: lowercased, URLs and HTML
stripped, contractions expanded, emoticons and punctuation removed, stop words dropped,
then POS-aware lemmatisation.

keep_negation (default True) holds "not", "no", "never" and the rest out of the stop word
list. Removing them would turn "not good" into "good", which is the wrong sentiment.

Usage:
    from clean_review import clean_review
    df["clean_text"] = df["review"].apply(clean_review)
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# These are stop words, so they would normally be deleted. keep_negation protects them.
NEGATION_WORDS = {
    "no", "nor", "not", "never", "neither", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without",
}
_FULL_STOPWORDS = set(stopwords.words("english"))
_STOP_KEEP_NEG = _FULL_STOPWORDS - NEGATION_WORDS
_STOP_TEXTBOOK = _FULL_STOPWORDS

_lemmatizer = WordNetLemmatizer()
_lemma_cache = {}   # 30k reviews repeat the same words constantly, so cache the lemmas

# Removed before the letters-only tokeniser, otherwise a face like ":D" leaves a stray
# "d" token behind. Written lowercase because the text is lowercased first.
_EMOTICONS = re.compile(
    r"(:-?\)|:-?\]|=\)|\(:|:-?d|=d|;-?\)|:'\)|:-?\(|:-?\[|=\(|\):|:-?/|:-?\\|:'\(|d:)"
)


def _penn_to_wordnet(tag):
    # WordNet wants a/v/r/n, NLTK's tagger gives Penn Treebank tags. Without this the
    # lemmatiser assumes everything is a noun and "better" never becomes "good".
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def _lemmatize(word, pos):
    key = (word, pos)
    if key not in _lemma_cache:
        _lemma_cache[key] = _lemmatizer.lemmatize(word, pos)
    return _lemma_cache[key]


def _expand_contractions(text):
    # Has to run before punctuation is stripped. Otherwise "wasn't" becomes "wasnt",
    # one meaningless token, and the negation is lost.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def clean_review(text, keep_negation=True):
    text = str(text).lower().replace("’", "'")
    # Scraped reviews carry these as literal two-character sequences, not real newlines.
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _expand_contractions(text)
    text = _EMOTICONS.sub(" ", text)

    # Letters only, which drops punctuation and numbers in one go.
    tokens = re.findall(r"[a-z]+", text)

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
