from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize


with open("Data_1.txt", "r") as f:
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