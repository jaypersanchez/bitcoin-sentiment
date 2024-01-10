import tweepy
from textblob import TextBlob
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can use os.getenv to get your environment variables
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def get_tweet_sentiment(tweet):
    analysis = TextBlob(tweet.lower())
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def fetch_tweets(query, max_tweets=20):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(max_tweets)
    return [tweet.text for tweet in tweets]

def append_to_json(file_path, data):
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Ensure the directory exists
output_directory = "data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# File path for the output file
output_file = os.path.join(output_directory, "alpaca-bitcoin-sentiment-dataset.json")

# List of keywords for extraction
keywords = ["bitcoin", "ethereum", "crypto", "xrp"]  # You can add more keywords here

# Process tweets for each keyword
for keyword in keywords:
    keyword_tweets = fetch_tweets(keyword, max_tweets=20)
    for tweet in keyword_tweets:
        sentiment = get_tweet_sentiment(tweet)
        tweet_data = {
            "instruction": "Detect the sentiment of the tweet.",
            "input": tweet,
            "output": sentiment,
            "date": tweet.created_at.strftime("%Y-%m-%d"),
            "time": tweet.created_at.strftime("%H:%M:%S")
        }
        append_to_json("crypto_tweets.json", tweet_data)
