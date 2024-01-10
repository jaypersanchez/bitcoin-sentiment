import json
import psycopg2
import torch
import numpy as np
import os
import faiss
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Environment variables
consumer_key = os.getenv('CONSUMER_KEY')
_dbname = os.getenv('DBNAME')
_user = os.getenv('POSTGRES_USER')
_password = os.getenv('POSTGRES_PASSWORD')
_host = os.getenv('POSTGRES_HOST')

# Initialize NLP model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

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

# Fetch and display corresponding tweet data
for tweet_id in result_ids:
    # Convert numpy.int32 to Python int
    tweet_id_native = int(tweet_id)
    cur.execute("SELECT input, output FROM tweet_sentiments WHERE id = %s", (tweet_id_native,))
    tweet = cur.fetchone()
    print(f"Tweet: {tweet[0]}, Sentiment: {tweet[1]}")

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()
