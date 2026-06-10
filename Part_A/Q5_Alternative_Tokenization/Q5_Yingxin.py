from textblob import TextBlob
import nltk
from pathlib import Path
from nltk.tokenize import word_tokenize


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "Data_1.txt"


with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# NLTK
nltk_tokens = word_tokenize(text)
print("NLTK Tokenizer")
print("============================================================")
print(nltk_tokens)

# TextBlob 
blob = TextBlob(text)
textblob_tokens = blob.words
print("\nTextBlob Tokenizer (Alternative)")
print("============================================================")
print(textblob_tokens)
