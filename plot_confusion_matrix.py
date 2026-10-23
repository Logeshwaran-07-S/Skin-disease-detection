import json
import matplotlib.pyplot as plt
import numpy as np

MATRIX_PATH = "model/confusion_matrix.csv"
CLASS_PATH = "model/class_names.json"
OUTPUT_IMAGE = "model/confusion_matrix.png"

# Load class names
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Load confusion matrix
matrix = np.loadtxt(MATRIX_PATH, delimiter=",", skiprows=1)

# Plot
plt.figure(figsize=(14, 12))
plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Skin Disease CNN (22 Classes)", fontsize=16)
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90, fontsize=9)
plt.yticks(tick_marks, class_names, fontsize=9)

plt.xlabel("Predicted Disease", fontsize=12)
plt.ylabel("Actual Disease", fontsize=12)
plt.tight_layout()

# Save image
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.close()

print(f"Confusion matrix plot successfully saved to: {OUTPUT_IMAGE}")