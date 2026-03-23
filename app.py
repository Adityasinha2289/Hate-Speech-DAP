import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- Text Preprocessing Function ---
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

# Define colors for each class
class_colors = {
    0: "#dc2626",  # Red for hate speech
    1: "#ca8a04",  # Yellow for offensive
    2: "#16a34a",  # Green for normal
}

def classify_text(message):
    """
    Classifies a message and returns the prediction with confidence score.
    """
    if not message or not message.strip():
        return "Please enter a message to classify.", None
    
    # Clean the text
    cleaned_message = clean_text(message)
    
    if not cleaned_message:
        return "Message is empty after preprocessing.", None
    
    try:
        # Get prediction from the model
        prediction = pipeline.predict([cleaned_message])[0]
        
        # Get confidence score (probabilities)
        probabilities = pipeline.predict_proba([cleaned_message])[0]
        confidence = float(max(probabilities))
        
        # Get the human-readable label
        label = class_labels.get(int(prediction), "Unknown")
        
        # Format the result
        if prediction == 0:  # Hate Speech
            result = f"🚨 **HATE SPEECH DETECTED**\n\n**Classification:** {label}\n\n**Confidence:** {confidence*100:.2f}%\n\n**Status:** This content is flagged as highly violative and should be removed."
        elif prediction == 1:  # Offensive
            result = f"⚠️ **OFFENSIVE CONTENT DETECTED**\n\n**Classification:** {label}\n\n**Confidence:** {confidence*100:.2f}%\n\n**Status:** This content contains offensive language and may need moderation."
        else:  # Normal
            result = f"✅ **CONTENT APPROVED**\n\n**Classification:** {label}\n\n**Confidence:** {confidence*100:.2f}%\n\n**Status:** This content appears to be normal and non-violative."
        
        return result, label
        
    except Exception as e:
        return f"Error during classification: {str(e)}", None

# --- Create Gradio Interface ---
with gr.Blocks(title="Hate Speech Detection Bot", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("""
    # Content Moderation Bot
    
    **Classify text as Hate Speech, Offensive Language, or Normal Speech**
    
    This AI-powered content moderation system uses machine learning to automatically detect harmful content.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            message_input = gr.Textbox(
                label="Enter Message to Classify",
                placeholder="Type a message here...",
                lines=3,
                interactive=True
            )
            
            classify_button = gr.Button("Analyze", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            result_output = gr.Markdown(
                value="Results will appear here after you submit a message.",
                label="Classification Result"
            )
            
            label_output = gr.Textbox(
                label="Classification Label",
                interactive=False,
                visible=False
            )
    
    gr.Markdown("""
    ---
    
    ### How It Works
    - **Hate Speech:** Content that attacks, demeans, or incites violence against individuals or groups
    - **Offensive Language:** Content with profanity, rudeness, or disrespect but not targeting specific groups
    - **Normal Speech:** Regular, non-violative content
    
    *Note: This classifier is for demonstration purposes. Always review results manually.*
    """)
    
    # Link button to classification function
    classify_button.click(
        fn=classify_text,
        inputs=[message_input],
        outputs=[result_output, label_output]
    )
    
    # Also classify on Enter key in textbox
    message_input.submit(
        fn=classify_text,
        inputs=[message_input],
        outputs=[result_output, label_output]
    )

if __name__ == "__main__":
    demo.launch()
