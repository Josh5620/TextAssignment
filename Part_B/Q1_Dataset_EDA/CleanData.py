import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


PART_B_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PART_B_DIR / "data" / "cyberbullying_tweets.csv"
CLEAN_DATA_FILE = PART_B_DIR / "data" / "cyberbullying_clean.csv"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)                   # remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)              # strip # keep word
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)        # sooooo -> soo
    text = re.sub(r'[^a-z\s]', '', text)               # strip non-alpha
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]   # stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]    # lemmatize
    return ' '.join(tokens)

# --- Load and clean dataset ---
df = pd.read_csv(RAW_DATA_FILE)
df['clean'] = df['tweet_text'].apply(clean_tweet)


# Drop rows where cleaning left nothing
df = df.dropna(subset=['clean'])
df = df[df['clean'].str.strip() != '']


CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_DATA_FILE, index=False)




