import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
df = pd.read_csv('cyberbullying_tweets.csv')
df['clean'] = df['tweet_text'].apply(clean_tweet)


# Drop rows where cleaning left nothing
df = df.dropna(subset=['clean'])
df = df[df['clean'].str.strip() != '']


df.to_csv('cyberbullying_clean.csv', index=False)




