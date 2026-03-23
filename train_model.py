import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# --- Text Preprocessing Function ---
# We need this to be identical in both train_model.py and app.py
try:
    # Attempt to load stop words, download if not found
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans text for model training and inference.
    - Lowercase
    - Remove URLs, mentions, hashtags, and punctuation
    - Remove stopwords
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (@...) and hashtags (#...)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text.strip()

# --- Main Training Process ---
if __name__ == "__main__":
    
    # 1. Load Data
    print("Loading labeled_data.csv...")
    try:
        df = pd.read_csv('labeled_data.csv')
    except FileNotFoundError:
        print("\n" + "="*50)
        print("ERROR: 'labeled_data.csv' not found!")
        print("Please download the dataset and place it in the same folder as this script.")
        print("="*50 + "\n")
        exit()
        
    # We only need the 'class' and 'tweet' columns
    df = df[['class', 'tweet']]

    # 2. Clean and Prepare Data
    print("Cleaning and preprocessing text...")
    # Apply the cleaning function to the 'tweet' column
    df['tweet'] = df['tweet'].apply(clean_text)

    # Define our features (X) and target (y)
    X = df['tweet']
    y = df['class']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: {len(y_train)} training samples, {len(y_test)} testing samples.")

    # 4. Create a Model Pipeline
    print("Building model pipeline...")
    # We create a pipeline that will:
    # 1. Vectorize the text using TF-IDF
    # 2. Train a Logistic Regression classifier
    #
    # We use class_weight='balanced' to automatically handle the
    # severe class imbalance in your dataset.
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # Now uses both single words and pairs
        ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42))
    ])

    # 5. Train the Model
    print("Training model... (This may take a minute)")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate the Model
    print("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['0: Hate Speech', '1: Offensive', '2: Normal']))
    print("-----------------------------\n")

    # 7. Save the Model
    model_filename = 'model.joblib'
    print(f"Saving trained model to {model_filename}...")
    joblib.dump(pipeline, model_filename)

    print("\nTraining complete! Your model is ready.")
    # --- Update this section in your train_model.py file ---


