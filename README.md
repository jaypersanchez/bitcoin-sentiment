# Run Instructions
1. Run data_preparation against data/alpaca-bitcoin-sentiment-dataset.json.  This will generate a file called cleaned_sentiment_dataset.json
2. Run data_processing_test.py.  This will clean the data and remove any special characters as well as resolved any mismatched
3.  This will then generate two files sentiment_model.pkl and vectorizer.pkl
4. Copy the cleaned_sentiment_dataset.json and both the sentiment_model.pkl and vectorizer.pkl into the project vfinancials-flask data directory