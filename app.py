from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- Initialize Flask App and CORS ---
app = Flask(__name__)
# CORS is required to allow your HTML file (on a 'file://' URL)
# to make requests to this server (on 'http://127.0.0.1')
CORS(app) 

# --- Text Preprocessing Function ---
# This MUST be identical to the one used in train_model.py
try:
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
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

# --- Load Trained Model ---
print("Loading trained model (model.joblib)...")
try:
    pipeline = joblib.load('model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("\n" + "="*50)
    print("ERROR: 'model.joblib' not found!")
    print("Please run `python3 train_model.py` first to train and save the model.")
    print("="*50 + "\n")
    exit()

# Define the human-readable labels for our classes
class_labels = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Normal Speech"
}

# --- Define API Endpoint ---
@app.route('/classify', methods=['POST'])
def classify_message():
    """
    The main API endpoint.
    Receives JSON data: {"message": "some text..."}
    Returns JSON data: {"class": 0, "label": "Hate Speech", "confidence": 0.9}
    """
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "No 'message' field in JSON payload."}), 400

    # 1. Get text from the incoming request
    message = request.json['message']
    
    # 2. Clean the text (must use the *exact* same function as training)
    cleaned_message = clean_text(message)
    
    # 3. Get prediction from the model
    # We pass the cleaned message as a list
    try:
        prediction = pipeline.predict([cleaned_message])[0]
        
        # 4. Get confidence score (probabilities)
        probabilities = pipeline.predict_proba([cleaned_message])[0]
        confidence = float(max(probabilities))
        
        # 5. Get the human-readable label
        label = class_labels.get(int(prediction), "Unknown")
        
        # 6. Format and send the response
        response = {
            "class": int(prediction),
            "label": label,
            "confidence": confidence
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run the Server ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)
