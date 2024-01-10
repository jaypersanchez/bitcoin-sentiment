import json
import psycopg2
import torch
import numpy as np
import os
import faiss
import time
import sys
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, set_seed

#Environment variables
consumer_key = os.getenv('CONSUMER_KEY')
_dbname = os.getenv('DBNAME')
_user = os.getenv('POSTGRES_USER')
_password = os.getenv('POSTGRES_PASSWORD')
_host = os.getenv('POSTGRES_HOST')

# Initialize NLP model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# Initialize a text generation pipeline with GPT model
generator = pipeline('text-generation', model='gpt2')  # Replace 'gpt2' with 'EleutherAI/gpt-neo-2.7B' or GPT-3 API endpoint
set_seed(42)

# Function to generate a semantic vector from a text input
def get_vector(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[0][0].detach().numpy()

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname='semantic', user='postgres', password='Ph1LL!fe', host='localhost'
)
cur = conn.cursor()

# Accept query from console
query = input("Enter your query: ")
query_vec = get_vector(query)

# Fetch vectors from the database
cur.execute("SELECT id, vector FROM tweet_sentiments")
rows = cur.fetchall()

# Prepare data for Faiss
ids = np.array([row[0] for row in rows])
embeddings = np.array([row[1] for row in rows])

# Dimension of the vectors
d = len(embeddings[0])

# Create a Faiss index
index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity

# Add vectors to the index
index.add(embeddings)

# Function to perform a search
def search(query_vec, k=5):
    _, indices = index.search(np.array([query_vec]), k)
    return ids[indices[0]]

result_ids = search(query_vec)
tweets_data = []
all_responses = []

# Fetch and display corresponding tweet data
for tweet_id in result_ids:
    # Convert numpy.int32 to Python int
    tweet_id_native = int(tweet_id)
    cur.execute("SELECT input, output FROM tweet_sentiments WHERE id = %s", (tweet_id_native,))
    tweet = cur.fetchone()
    #print(f"Tweet: {tweet[0]}, Sentiment: {tweet[1]}")
    tweets_data.append(tweet)

# Initialize sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]

def generate_response(input_text):
    generator = pipeline('text-generation', model='gpt2')  # or another model of your choice
    set_seed(42)
    #prompt = f"Based on this tweet: '{input_text}', a suitable response would be: "
    prompt = f"{input_text}"
    response = generator(prompt, max_length=50)
    return response[0]['generated_text']

# Generate and print responses for each tweet
for tweet in tweets_data:
    tweet_content = tweet[0]  # Assuming tweet[0] is the tweet text
    response_text = generate_response(tweet_content)
    
    # Analyze sentiment of the response
    sentiment_result = analyze_sentiment(response_text)
    all_responses.append(sentiment_result)
    
    #print(f"Tweet: {tweet_content}, Generated Response: {response_text}")
    #print(f"{response_text}")
    # Print a new line
    print("\n")

    # Print the response text with a typing effect
    for char in response_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)  # Adjust this value to speed up or slow down the effect
    print(f"\nSentiment: {sentiment_result['label']} (Score: {sentiment_result['score']:.2f})\n")

    # Print another new line for spacing between tweets
    print("\n")

# Aggregate the overall sentiment
positive = sum(1 for resp in all_responses if resp['label'] == 'POSITIVE')
negative = sum(1 for resp in all_responses if resp['label'] == 'NEGATIVE')

if positive > negative:
    overall_sentiment = "Positive"
elif negative > positive:
    overall_sentiment = "Negative"
else:
    overall_sentiment = "Neutral"

print(f"Overall Sentiment of Responses: {overall_sentiment}")


# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()
