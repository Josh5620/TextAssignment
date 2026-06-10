import spacy
import nltk
from pathlib import Path
from nltk.tokenize import word_tokenize


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Data_1.txt"


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load text data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# NLTK Tokenizer Q1
nltk_tokens = word_tokenize(text)
print("=== NLTK Tokenizer (Q1) ===")
print(nltk_tokens)

# SpaCy Tokenizer Q5
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("\n=== SpaCy Tokenizer (Q5 Alternative) ===")
print(spacy_tokens)
