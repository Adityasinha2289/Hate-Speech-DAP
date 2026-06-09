from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import joblib
import nltk
import re
from nltk.corpus import stopwords

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "hate_speech_detector.html"

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])
    return text.strip()


print("Loading trained model (model.joblib)...")
PIPELINE = joblib.load(ROOT / "model.joblib")
print("Model loaded successfully!")

CLASS_LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Normal Speech",
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, "text/plain; charset=utf-8", b"")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())
            return
        if self.path in ("/classify", "/api/classify"):
            self._send(200, "application/json; charset=utf-8", b'{"status":"ok"}')
            return
        self._send(404, "text/plain; charset=utf-8", b"Not Found")

    def do_POST(self):
        if self.path not in ("/classify", "/api/classify"):
            self._send(404, "application/json; charset=utf-8", b'{"error":"Not Found"}')
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = payload.get("message", "")

            if not message or not str(message).strip():
                response = {"class": -1, "message": "Please enter a message to classify."}
                self._send(400, "application/json; charset=utf-8", json.dumps(response).encode("utf-8"))
                return

            cleaned_message = clean_text(message)
            if not cleaned_message:
                response = {"class": -1, "message": "Message is empty after preprocessing."}
                self._send(400, "application/json; charset=utf-8", json.dumps(response).encode("utf-8"))
                return

            prediction = int(PIPELINE.predict([cleaned_message])[0])
            probabilities = PIPELINE.predict_proba([cleaned_message])[0]
            confidence = float(max(probabilities))

            response = {
                "class": prediction,
                "label": CLASS_LABELS.get(prediction, "Unknown"),
                "confidence": confidence,
                "message": message,
            }
            self._send(200, "application/json; charset=utf-8", json.dumps(response).encode("utf-8"))
        except Exception as exc:
            response = {"class": -1, "message": f"Server error: {exc}"}
            self._send(500, "application/json; charset=utf-8", json.dumps(response).encode("utf-8"))


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 7861), Handler)
    print("Serving frontend on http://127.0.0.1:7861")
    server.serve_forever()


if __name__ == "__main__":
    main()
