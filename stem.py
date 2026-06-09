import nltk
from nltk.stem import RegexpStemmer, PorterStemmer, LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Load text data
with open("Data_1.txt", "r") as f:
    text = f.read()

# Extract first 2 sentences and tokenize
sentences = sent_tokenize(text)[:1]
words = word_tokenize(" ".join(sentences))

# Regular Expression Stemmer
reg_stemmer = RegexpStemmer('ing$|ed$|s$|ness$', min=4)
print("=== Regular Expression Stemmer ===")
for word in words:
    print(f"{word} -> {reg_stemmer.stem(word)}")

# Porter Stemmer
porter = PorterStemmer()
print("\n=== Porter Stemmer ===")
for word in words:
    print(f"{word} -> {porter.stem(word)}")

# Lancaster Stemmer
lancaster = LancasterStemmer()
print("\n=== Lancaster Stemmer ===")
for word in words:
    print(f"{word} -> {lancaster.stem(word)}")

