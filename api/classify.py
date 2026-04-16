from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = joblib.load(ROOT / "model.joblib")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text.strip()


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = payload.get("message", "")

            if not message or not str(message).strip():
                self._send(400, {"class": -1, "message": "Please enter a message to classify."})
                return

            cleaned_message = clean_text(message)
            if not cleaned_message:
                self._send(400, {"class": -1, "message": "Message is empty after preprocessing."})
                return

            prediction = int(PIPELINE.predict([cleaned_message])[0])
            probabilities = PIPELINE.predict_proba([cleaned_message])[0]
            confidence = float(max(probabilities))

            labels = {
                0: "Hate Speech",
                1: "Offensive Language",
                2: "Normal Speech",
            }

            self._send(
                200,
                {
                    "class": prediction,
                    "label": labels.get(prediction, "Unknown"),
                    "confidence": confidence,
                    "message": message,
                },
            )
        except Exception as exc:
            self._send(500, {"class": -1, "message": f"Server error: {exc}"})
