import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data

# Assuming the JSON file is in the 'data' directory and named 'alpaca-bitcoin-sentiment-dataset.json'
file_path = 'data/alpaca-bitcoin-sentiment-dataset.json'

# Print a loading message
print("Loading the dataset. Please wait.....")
# Load the dataset into a Pandas DataFrame
try:
    df = pd.read_json(file_path)
    # Display the first few rows to verify the data
    print("Printing Dataframe of source data")
    print(df.head())
     # Call the preprocess_data function from data_processing.py
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # Print a success message
    print(f"Dataset loaded successfully! Number of rows: {len(df)}")
    
except Exception as e:
    print(f"Data Load Error loading the dataset: {e}")
