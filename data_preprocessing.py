# data_preprocessing.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    
    try:
        print("Preparing features and labels....")
        # X contains the features, y contains the target variable (labels)
        X = df.drop('input', axis=1)
        y = df['input']
        print(df.head())
        
        print("Splitting dataset....")
        # Split the dataset: 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Print the shapes to verify the split
        print(f"Training set shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

        # Return the processed data
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Preprocessing Error loading the dataset: {e}")
        return None

# If you want to test this module separately
if __name__ == "__main__":
    # Assuming 'df' is your DataFrame
    # You should replace this with the actual loading of your dataset
    df = pd.read_

    # Call the preprocess_data function
    X_train, X_test, y_train, y_test = preprocess_data(df)
