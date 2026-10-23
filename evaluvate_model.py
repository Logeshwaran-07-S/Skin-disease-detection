
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

TEST_PATH = "Dataset/Test"
MODEL_PATH = "model/skin_disease_cnn.keras"
CLASS_PATH = "model/class_names.json"
REPORT_PATH = "model/evaluation_report.txt"
MATRIX_PATH = "model/confusion_matrix.csv"

# Load saved model and class names
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Load test images without shuffling
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

# Collect actual labels and model predictions
actual_labels = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

predicted_scores = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(predicted_scores, axis=1)

# Create a report for every class
report = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

accuracy = np.mean(actual_labels == predicted_labels)

print("\nOverall test accuracy:", f"{accuracy * 100:.2f}%")
print("\nClass-wise evaluation:\n")
print(report)

# Save the report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Overall test accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)

# Save the confusion matrix
matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

np.savetxt(
    MATRIX_PATH,
    matrix,
    delimiter=",",
    fmt="%d",
    header="Rows=actual classes; columns=predicted classes",
    comments=""
)

print("\nReport saved to:", REPORT_PATH)
print("Confusion matrix saved to:", MATRIX_PATH)
