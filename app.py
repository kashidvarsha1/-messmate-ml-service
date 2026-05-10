
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

print("Loading model v3...")
classifier = pipeline(
    "text-classification",
    model="varshakashid/messmate-review-detector",
    tokenizer="varshakashid/messmate-review-detector",
    device=-1
)
print("Model v3 loaded - 91.95% accuracy!")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = classifier(text)[0]
    return jsonify({
        "label": result["label"],
        "confidence": round(result["score"] * 100, 2),
        "isFake": result["label"] == "fake",
        "sentiment": "negative" if result["label"] == "fake" else "positive"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "v3", "accuracy": "91.95%"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
