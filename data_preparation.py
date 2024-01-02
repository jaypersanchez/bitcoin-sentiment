import json
import pandas as pd

# Load your dataset in JSON format (replace 'your_dataset.json' with your actual dataset file)
with open('data/alpaca-bitcoin-sentiment-dataset.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Convert JSON data to a DataFrame
df = pd.DataFrame(data)

# Step 1: Correct Labels
# Ensure that your labels match the expected classes ('Positive', 'Negative', 'Neutral')
# If there are any labels that don't match, correct them
df['output'] = df['output'].replace({'pos': 'Positive', 'neg': 'Negative', 'neu': 'Neutral'})

# Step 2: Check Missing Labels
# Check for any missing labels in your dataset
missing_labels = df[df['output'].isnull()]
if not missing_labels.empty:
    # You can either drop rows with missing labels or fill them with a default value
    df.dropna(subset=['output'], inplace=True)  # Drop rows with missing labels
    # Alternatively, you can fill missing labels with a default class:
    # df['output'].fillna('Neutral', inplace=True)

# Step 3: Examine Sentiment Distribution
# Examine the distribution of sentiments to check for class imbalance
sentiment_counts = df['output'].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)

# Optionally, you can perform oversampling, undersampling, or use different evaluation metrics
# to handle class imbalance based on the distribution.

# Save the cleaned DataFrame to a new JSON file
cleaned_file_path = 'data/cleaned_sentiment_dataset.json'  # Replace with your desired file path
#df.to_json(cleaned_file_path, orient='records', lines=True, force_ascii=False)
# Convert DataFrame to a list of JSON records (objects)
json_records = df.to_dict(orient='records')

# Save the list of JSON records as a JSON array
with open(cleaned_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_records, json_file, ensure_ascii=False, indent=4)


print(f"Cleaned data saved to {cleaned_file_path}")
