
import sys
import json
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (128, 128)
MODEL_PATH = "model/skin_disease_cnn.keras"
CLASS_PATH = "model/class_names.json"


def predict_image(image_path):
    # Load the saved model and class labels
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    # Load the image and resize it to match training
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # The model includes its own Rescaling layer
    predictions = model.predict(image_array, verbose=0)[0]

    # Show the three highest-scoring classes
    top_indices = np.argsort(predictions)[-3:][::-1]

    print("\nPrediction results")
    print("------------------")

    for index in top_indices:
        print(
            f"{class_names[index]}: "
            f"{predictions[index] * 100:.2f}%"
        )

    print(
        "\nNote: These are model scores, not a medical diagnosis. "
        "Consult a qualified healthcare professional for assessment."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python predict_image.py "path\\to\\image.jpg"'
        )
        sys.exit(1)

    predict_image(sys.argv[1])
