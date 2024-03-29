from data_preprocessing import preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import issparse
import re
import pickle
import os
print("Current Working Directory:", os.getcwd())

def preprocess_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)
    # Remove new lines and other escape sequences
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', '')
    # Optional: additional preprocessing steps like lowercasing, etc.
    return text

# Assuming the JSON file is in the 'data' directory and named 'your_dataset.json'
file_path = 'data/cleaned_sentiment_dataset.json'
# Print a loading message
print("Loading the dataset for testing. Please wait.....")
df = pd.read_json(file_path)

# Call the preprocess_data function
data = preprocess_data(df)

# Call the preprocess_data function
#data = preprocess_data(df)
print("Data after preprocessing:", len(data[0]), len(data[1]), len(data[2]), len(data[3]))  # Check after preprocessing

def check_sample_lengths(X, y, mismatch_file='mismatch_data.txt'):
    if X.shape[0] != len(y):
        with open(mismatch_file, 'a') as file:
            file.write("Warning: Mismatch in number of samples between features and labels.\n")
            file.write(f"Length of features: {X.shape[0]}\n")
            file.write(f"Length of labels: {len(y)}\n")

            # Append the first few rows of X and y if there's a mismatch
            file.write("First few rows of features (X):\n")
            if issparse(X):
                # Convert to dense format for appending if X is a sparse matrix
                file.write(str(X[:5].toarray()) + '\n')
            else:
                file.write(str(X[:5]) + '\n')

            file.write("First few rows of labels (y):\n")
            file.write(str(y[:5]) + '\n')

        print("Warning: Logged mismatched sample sizes to", mismatch_file)


def check_and_log_mismatch(X, y, mismatch_file='mismatch_data.txt'):
    if len(X) != len(y):
        print("Mismatch detected. Excluding problematic rows and logging details.")
        min_length = min(len(X), len(y))
        mismatched_indices = list(range(min_length, max(len(X), len(y))))

        with open(mismatch_file, 'a', encoding='utf-8') as file:
            file.write("Mismatched Data:\n")
            for index in mismatched_indices:
                if index < len(X):
                    file.write(f"X[{index}]: {X.iloc[index]}\n")
                if index < len(y):
                    file.write(f"y[{index}]: {y.iloc[index]}\n")

        # Exclude mismatched rows
        X = X[:min_length]
        y = y[:min_length]
        return X, y, True
    else:
        print("No mismatch detected.")
        return X, y, False

def save_model_and_vectorizer(model, vectorizer, model_file_name='data/sentiment_model.pkl', vectorizer_file_name='data/vectorizer.pkl'):
    try:
        # Check if the vectorizer is fitted
        if not vectorizer.fixed_vocabulary_:
            print("Vectorizer is not fitted. Fitting it now...")
            # Assuming you have X_train, which contains the training data
            vectorizer.fit(X_train)
            
        # Save the model
        with open(model_file_name, 'wb') as model_file:
            pickle.dump(model, model_file)
        
        # Save the vectorizer
        with open(vectorizer_file_name, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        
        print("Model and vectorizer saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

def remove_mismatched_data(X, y):
    if len(X) != len(y):
        print("Mismatch detected. Excluding problematic rows.")
        min_length = min(len(X), len(y))

        # Exclude mismatched rows
        X = X[:min_length]
        y = y[:min_length]
        return X, y
    else:
        print("No mismatch detected.")
        return X, y
        
# Use the processed data as needed
if data:
    X_train, X_test, y_train, y_test = data
    
    # Apply preprocess_tweet to each tweet in X_train and X_test
    X_train = [preprocess_tweet(tweet) for tweet in X_train]
    X_test = [preprocess_tweet(tweet) for tweet in X_test]
    
    # Vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer()
    
    # Check for mismatch and handle it
    X_train, y_train, mismatch_found_train = check_and_log_mismatch(X_train, y_train)
    X_test, y_test, mismatch_found_test = check_and_log_mismatch(X_test, y_test)

    # Remove mismatched data
    if mismatch_found_train:
        X_train, y_train = remove_mismatched_data(X_train, y_train)
    if mismatch_found_test:
        X_test, y_test = remove_mismatched_data(X_test, y_test)

    #if not mismatch_found:
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    print("X_train_vectorized shape:", X_train_vectorized.shape)
    print("y_train shape:", y_train.shape)

    # Check if the lengths of X_train_vectorized and y_train match
    check_sample_lengths(X_train_vectorized, y_train)
        
    # Ensure y_train aligns with X_train_vectorized
    if X_train_vectorized.shape[0] != len(y_train):
        print("Adjusting y_train to match X_train_vectorized...")
        y_train = y_train.iloc[:X_train_vectorized.shape[0]]
            
    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vectorized, y_train)
        
    # Make predictions on the test set
    y_pred = nb_classifier.predict(X_test_vectorized)
        
    # Align y_test with y_pred
    if len(y_pred) != len(y_test):
        print("Adjusting y_test to match the predictions...")
        y_test = y_test.iloc[:len(y_pred)]
        
    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
        
    # Display the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_rep)
    
    # Save the trained model and vectorizer
    save_model_and_vectorizer(nb_classifier, vectorizer)
