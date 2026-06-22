"""
clean_review_v2 — improved text cleaner for the Part B sentiment pipeline.

This EXTENDS Christabel's TripAdvisor cleaner (CleanTripAdvisorData.py).
Her original already (a) keeps negation words and (b) expands contractions
so that "not" survives. This version adds the techniques testing showed help,
plus fixes that testing showed were needed:

  + negation marking : after a negator, prefix the next few words with "neg_"
                       so "not good" -> "not neg_good" (a different feature
                       from plain "good"). Classic sentiment technique.
  + POS lemmatization: reduce words to base form using their part-of-speech,
                       so verbs/adjectives lemmatize too (loved -> love,
                       disappointing -> disappoint, better -> good). Plain
                       WordNetLemmatizer assumes NOUN and leaves sentiment
                       words untouched, so POS tagging is required (W05 s22).
  + emoticon rescue  : map text emoticons to sentiment tokens BEFORE the
                       letters-only strip, so ":)" -> emojipos, ":(" -> emojineg.
                       (The Yelp data has 0% real emojis but ~4.3% emoticons,
                       and they carry sentiment.) Bare numbers carry no
                       standalone sentiment and are dropped.
  FIX cannot         : "cannot" is expanded to "can not" so its negation is
                       kept (plain "cannot" was slipping past the negator set).
  FIX scope cap      : the neg_ sticker only reaches the next 4 words, then
                       stops. Without this cap it over-tagged and hurt LR.
  FIX literal escapes: strip leftover "\\n" / "\\t" text codes that survive
                       in some scraped reviews (they were becoming junk tokens).

Tested on Yelp (5-class and 3-class) vs Christabel's original: the negation
handling was best or tied on every model/task, with the biggest gains on the
weaker Naive Bayes.

Usage:
    from clean_review_v2 import clean_review_v2
    df["clean_text"] = df["text"].apply(clean_review_v2)
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Reproducibility guard: make sure the NLTK data this cleaner needs is present
# (so it runs top-to-bottom on a fresh machine, e.g. the marker's). Each resource
# is tried independently so an id that doesn't exist on a given NLTK version
# cannot abort the whole run.
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

# Negation words to protect (same set as Christabel's cleaner).
NEGATION_WORDS = {
    "no", "nor", "not", "never", "neither", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without",
}
# Standard English stoplist, but KEEP the negation words above.
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()
_lemma_cache = {}

# How many words after a negator get the neg_ tag before the scope resets.
NEGATION_SCOPE = 4

# Emoticon -> sentiment token. Patterns are LOWERCASE because cleaning lowercases
# the text first (so ":D" is already ":d" by the time these run). The replacement
# tokens are letters-only so the [a-z]+ tokenizer keeps each as a single token.
_EMOTICON_POS = re.compile(r"(:-?\)|:-?\]|=\)|\(:|:-?d|=d|;-?\)|:'\))")
_EMOTICON_NEG = re.compile(r"(:-?\(|:-?\[|=\(|\):|:-?/|:-?\\|:'\(|d:)")


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
    # Expand the contractions that carry negation BEFORE punctuation is removed.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def clean_review_v2(text, scope=NEGATION_SCOPE):
    # 1. lowercase and normalise the curly apostrophe
    text = str(text).lower().replace("’", "'")
    # 2. strip leftover literal escape codes (\n \t \r) seen in scraped text
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    # 3. remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # 4. expand negation-bearing contractions so "not" is kept
    text = _expand_contractions(text)
    # 5. rescue emoticons into sentiment tokens before the letters-only strip
    #    (done after URL removal so "http://" slashes aren't seen as ":/")
    text = _EMOTICON_POS.sub(" emojipos ", text)
    text = _EMOTICON_NEG.sub(" emojineg ", text)

    # 6. tokenise keeping clause punctuation, so negation scope can be bounded
    tokens = re.findall(r"[a-z]+|[.!?,;]", text)

    # 7. negation marking with a capped scope window
    marked, since_negator = [], None     # None = not currently negating
    for tok in tokens:
        if tok in ".!?,;":               # clause boundary resets negation
            since_negator = None
            continue
        if tok in NEGATION_WORDS:         # start a new negation scope
            marked.append(tok)
            since_negator = 0
            continue
        if since_negator is not None and since_negator < scope:
            marked.append("neg_" + tok)
            since_negator += 1
        else:
            marked.append(tok)
            since_negator = None

    # 8. POS-tag the base words once (sequence context), then drop stopwords /
    #    single letters and lemmatize with the correct part-of-speech.
    bases = [tok[4:] if tok.startswith("neg_") else tok for tok in marked]
    tags = nltk.pos_tag(bases)
    cleaned = []
    for tok, (_word, tag) in zip(marked, tags):
        base = tok[4:] if tok.startswith("neg_") else tok
        if base in NEGATION_WORDS:
            cleaned.append(tok)
        elif base not in STOP_WORDS and len(base) > 1:
            lemma = _lemmatize(base, _penn_to_wordnet(tag))
            cleaned.append("neg_" + lemma if tok.startswith("neg_") else lemma)
    return " ".join(cleaned)


if __name__ == "__main__":
    samples = [
        "The food was not bad at all, but the service wasn't great.",
        "I do not like this place. It was good though.",
        "Absolutely loved it! Best meal ever, would definitely return. :)",
        "Cannot recommend this dump. Waited 30 mins for cold fries :(",
    ]
    for s in samples:
        print("RAW :", s)
        print("V2  :", clean_review_v2(s))
        print()
