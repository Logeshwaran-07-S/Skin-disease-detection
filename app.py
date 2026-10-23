"""
Flask backend for the AI Skin Disease Detection and Healthcare Chatbot.

Routes
  GET  /              old HTML page (templates/index.html), still works
  GET  /api/health    check that the server and model are ready
  GET  /api/classes   list of the 22 disease classes
  POST /api/predict   upload an image (form field "image") -> top-3 predictions
  POST /api/chat      {"message": "...", "disease": "<optional latest result>"} -> chatbot reply

The React frontend (frontend/) calls the /api/... routes.
"""

import os
import uuid

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from chatbot import get_reply, pretty
from predict_service import class_names, get_model_info, predict_skin_disease

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_MB = 8

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_prediction(file):
    """Validate, save temporarily, predict, delete. Returns (result, error)."""
    if file is None or file.filename == "":
        return None, "Please choose an image."
    if not allowed_file(file.filename):
        return None, "Only PNG, JPG, JPEG and WEBP images are allowed."

    ext = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file.save(save_path)
    try:
        result = predict_skin_disease(save_path)
        # add readable names, e.g. "DrugEruption" -> "Drug Eruption"
        result["predicted_label"] = pretty(result["predicted_disease"])
        for item in result["top_3"]:
            item["label"] = pretty(item["disease"])
        return result, None
    except Exception as exc:  # unreadable / corrupted image
        print("Prediction error:", exc)
        return None, "Could not read this image. Please upload a valid photo."
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# ---------------------------------------------------------------- old HTML page
@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        result, error = run_prediction(request.files.get("image"))
    return render_template("index.html", result=result, error=error)


# ---------------------------------------------------------------- JSON API
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", **get_model_info()})


@app.get("/api/classes")
def classes():
    return jsonify({"classes": [{"name": c, "label": pretty(c)} for c in class_names]})


@app.post("/api/predict")
def api_predict():
    result, error = run_prediction(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)


@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))[:500]
    disease = data.get("disease")
    return jsonify(get_reply(message, disease=disease, class_names=class_names))


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    msg = f"Image is too large. Maximum size is {MAX_MB} MB."
    if request.path.startswith("/api/"):
        return jsonify({"error": msg}), 413
    return render_template("index.html", result=None, error=msg), 413


if __name__ == "__main__":
    # debug=True is for development only; do not expose this server publicly.
    # use_reloader=False stops Flask from loading the TensorFlow model twice.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
