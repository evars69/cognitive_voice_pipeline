from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from src.predict import predict  # reuse existing prediction logic

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "MemoTag Voice API is running ✅"

@app.route("/predict", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        pred = predict(filepath)
        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("MemoTag Voice API is running locally on http://127.0.0.1:5000 ✅")
    app.run(debug=True, port=5000)
