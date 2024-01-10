import json
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print(torch.__version__)
# Load JSON data from a file
file_path = 'data/cleaned_sentiment_dataset.json'  # Update with the correct path
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

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
    dbname="semantic", user="postgres", password="Ph1LL!fe", host="localhost"
)
cur = conn.cursor()

# Create a table for storing vectors (if not already created)
cur.execute("""
    CREATE TABLE IF NOT EXISTS tweet_sentiments (
        id SERIAL PRIMARY KEY,
        instruction TEXT,
        input TEXT,
        output TEXT,
        vector float8[]
    )
""")

# Insert data into the table
for item in data:
    vec = get_vector(item['input'])
    # Convert numpy array to a list of Python floats
    vec_list = vec.tolist()
    cur.execute(
        "INSERT INTO tweet_sentiments (instruction, input, output, vector) VALUES (%s, %s, %s, %s)",
        (item['instruction'], item['input'], item['output'], vec_list)
    )

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()
