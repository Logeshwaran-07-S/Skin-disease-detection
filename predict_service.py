"""
Prediction service (Member 1).

Loads the trained model ONCE and returns the top-3 predicted skin diseases
for an image. Used by app.py (website + API) and can be run on its own to test.

It picks the best model available in the model/ folder:
    1. model/skin_disease_mobilenet.keras  (MobileNetV2, 224 x 224)  <- final model
    2. model/skin_disease_cnn.keras        (custom CNN, 128 x 128)    <- fallback
The input size is read from the model itself, so it always matches training.
"""

import json
import os

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "skin_disease_mobilenet.keras"),
    os.path.join(MODEL_DIR, "skin_disease_cnn.keras"),
]


def _find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained model found. Expected one of:\n  " + "\n  ".join(MODEL_CANDIDATES)
    )


MODEL_PATH = _find_model_path()
print(f"Loading model: {os.path.basename(MODEL_PATH)} ...")
model = tf.keras.models.load_model(MODEL_PATH)

# Read the image size the model was trained with, e.g. (None, 224, 224, 3)
_, IMG_HEIGHT, IMG_WIDTH, _ = model.input_shape
print(f"Model ready. Input size: {IMG_WIDTH} x {IMG_HEIGHT}")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


def get_model_info():
    """Small summary used by the /api/health endpoint."""
    return {
        "model_file": os.path.basename(MODEL_PATH),
        "input_size": [int(IMG_WIDTH), int(IMG_HEIGHT)],
        "num_classes": len(class_names),
    }


def predict_skin_disease(image_path):
    """
    Input:  image_path (str)
    Output: {
        "predicted_disease": str,
        "confidence": float (0-100),
        "top_3": [{"disease": str, "confidence": float}, ...]
    }
    Normalisation is done inside the model (Rescaling layer), so pixels
    are passed in the 0-255 range here.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_3_indices = np.argsort(predictions)[-3:][::-1]

    results = [
        {
            "disease": class_names[idx],
            "confidence": round(float(predictions[idx]) * 100, 2),
        }
        for idx in top_3_indices
    ]

    return {
        "predicted_disease": results[0]["disease"],
        "confidence": results[0]["confidence"],
        "top_3": results,
    }


if __name__ == "__main__":
    sample = os.path.join(BASE_DIR, "test_images", "sample.jpeg")
    print("\nService test output:")
    print(predict_skin_disease(sample))
