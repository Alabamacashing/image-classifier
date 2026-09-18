from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch
import json
import urllib.request
import io

app = Flask(__name__)

# Load ResNet-50 pretrained model
print("Loading image classification model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()
print("Model ready!")

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(LABELS_URL) as url:
    labels = json.loads(url.read().decode())

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            results.append({
                "label": labels[idx.item()].replace("_", " ").title(),
                "confidence": round(prob.item() * 100, 2)
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)