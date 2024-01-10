import json
import psycopg2
import torch
import numpy as np
import os
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Environment variables
consumer_key = os.getenv('CONSUMER_KEY')
dbname = os.getenv('DBNAME')
user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
host = os.getenv('POSTGRES_HOST')

# Initialize NLP model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Function to generate a semantic vector from a text input
def get_vector(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[0][0].detach().numpy()

def cosine_similarity(vec_a, vec_b):
    return 1 - cosine(vec_a, vec_b)

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=dbname, user=user, password=password, host=host
)
cur = conn.cursor()

# Accept query from console
query = input("Enter your query: ")
query_vec = get_vector(query).tolist()  # Convert to a list of floats

# Fetch vectors from the database
cur.execute("SELECT id, vector FROM tweet_sentiments")
rows = cur.fetchall()

# Calculate similarities
similarities = []
for row in rows:
    vec_b = np.array(row[1])
    similarity = cosine_similarity(np.array(query_vec), vec_b)
    similarities.append((row[0], similarity))

# Sort by highest similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Get top results (for example, top 5)
top_results = similarities[:5]

# Retrieve and display the results
for result in top_results:
    cur.execute("SELECT input, output FROM tweet_sentiments WHERE id = %s", (result[0],))
    tweet = cur.fetchone()
    print(f"Tweet: {tweet[0]}, Sentiment: {tweet[1]}, Similarity: {result[1]}")

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()
