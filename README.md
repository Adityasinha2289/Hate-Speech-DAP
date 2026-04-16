# Hate Speech DAP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge" />
  <img src="https://img.shields.io/badge/Frontend-HTML%20%2B%20Tailwind-38BDF8?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Frontend badge" />
  <img src="https://img.shields.io/badge/Model-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Model badge" />
</p>

## What This Project Does

Hate Speech DAP is a local content moderation experience that classifies text into:

- `Hate Speech`
- `Offensive Language`
- `Normal Speech`

It uses a trained machine learning pipeline with:

- TF-IDF text features
- Bigram support
- Logistic Regression classification
- Balanced class weighting

The project includes a dramatic, product-style frontend so the app feels more like a real moderation dashboard instead of a basic demo.

## Highlights

- Real-time classification through a local `/classify` endpoint
- Custom frontend with `Workspace`, `Dashboard`, `Analytics`, `Model Config`, `Policy Rules`, and `Settings`
- Live counters and analytics that update as predictions are made
- Clean preprocessing for URLs, mentions, hashtags, punctuation, and stopwords
- Trained model artifact saved as `model.joblib`
- Dataset file included as `labeled_data.csv`

## Project Structure

```text
.
├── index.html
├── api/
│   └── classify.py
├── app.py
├── server.py
├── hate_speech_detector.html
├── train_model.py
├── model.joblib
├── labeled_data.csv
├── requirements.txt
├── vercel.json
├── Dockerfile
└── README.md
```

## How It Works

1. A user types a message into the frontend.
2. The frontend sends the message to the local Python backend.
3. The backend cleans the text using the same preprocessing logic used during training.
4. The trained model predicts one of the three moderation classes.
5. The UI renders the result with confidence and updates the dashboard views.

## Tech Stack

- Python
- scikit-learn
- joblib
- nltk
- Python runtime on Vercel
- HTML, CSS, JavaScript
- Tailwind CSS CDN for styling support

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Adityasinha2289/Hate-Speech-DAP.git
cd Hate-Speech-DAP
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Vercel-friendly API locally

```bash
python3 server.py
```

### 5. Open the app

Visit:

```text
http://127.0.0.1:7861
```

### Local demo stack

If you want to run the original richer local demo server, install the extra UI dependencies:

```bash
pip install gradio nltk flask flask-cors pandas
python3 app.py
```

## Free Public Hosting

### Option 1: Vercel

This repo is now Vercel-ready:

- `index.html` is the public entry point
- `api/classify.py` serves the moderation API at `/api/classify`
- `vercel.json` keeps the function bundle lean

To publish it on Vercel:

1. Sign in to Vercel with GitHub.
2. Import this repository.
3. Deploy it as a project.
4. Open the generated public `vercel.app` link.

### Option 2: Hugging Face Spaces

If you want a model-demo style link, you can also deploy the repo as a Docker Space.

## Vercel Notes

- The frontend calls the API with a relative `/api/classify` path.
- The model is loaded directly from `model.joblib`.
- The function uses lightweight built-in preprocessing so it can run without downloading NLTK data at runtime.
- The repo stays free-tier friendly by ignoring local-only training files in `.vercelignore`.

## Model Details

- Training data: `labeled_data.csv`
- Text preprocessing:
  - lowercase conversion
  - URL removal
  - mention and hashtag removal
  - punctuation removal
  - stopword removal
- Feature extraction: `TfidfVectorizer(ngram_range=(1, 2))`
- Classifier: `LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)`

## API

### `POST /classify`

Request body:

```json
{
  "message": "your text here"
}
```

Response:

```json
{
  "class": 2,
  "label": "Normal Speech",
  "confidence": 0.59,
  "message": "your text here"
}
```

## Notes

- The project is designed for demonstration and experimentation.
- Human review is still important for moderation decisions.
- The current frontend is local-first and uses the same-origin API route for fast response and a clean setup.

## Created By

Created by **Aditya Sinha**

## License

MIT License
