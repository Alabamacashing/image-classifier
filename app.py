from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

def classify_image(image_bytes):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"})

    try:
        image_bytes = file.read()
        results = classify_image(image_bytes)

        if isinstance(results, dict) and "error" in results:
            return jsonify({"error": results["error"]})

        # Format top 5 predictions
        predictions = []
        for item in results[:5]:
            predictions.append({
                "label": item["label"].replace("_", " ").title(),
                "confidence": round(item["score"] * 100, 2)
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)