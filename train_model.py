import os
import json
import tensorflow as tf

# -----------------------------
# 1. Basic settings
# -----------------------------
IMAGE_SIZE = (224, 224)          # MobileNetV2 works best at 224x224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10               # Phase 1: train only the new top layers
EPOCHS_FINETUNE = 10             # Phase 2: fine-tune part of MobileNetV2
FINE_TUNE_LAYERS = 40            # How many of the last MobileNetV2 layers to unfreeze

TRAIN_PATH = "Dataset/Train"
TEST_PATH = "Dataset/Test"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "skin_disease_mobilenet.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# -----------------------------
# 2. Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    validation_split=0.1, subset="training", seed=123, label_mode="int"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    validation_split=0.1, subset="validation", seed=123, label_mode="int"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    shuffle=False, label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)

print("Number of classes:", num_classes)

# -----------------------------
# 3. Class weights (helps with classes that have fewer images)
# -----------------------------
counts = []
for name in class_names:
    folder = os.path.join(TRAIN_PATH, name)
    counts.append(len([f for f in os.listdir(folder)
                       if os.path.isfile(os.path.join(folder, f))]))
total = sum(counts)
class_weight = {i: total / (num_classes * c) for i, c in enumerate(counts) if c > 0}
print("Images per class:", dict(zip(class_names, counts)))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# 4. Build the model (MobileNetV2 transfer learning)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name="augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"            # downloads once (~14 MB), needs internet the first time
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)                              # only active during training
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # MobileNetV2 expects [-1, 1]
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

metrics = [
    "accuracy",
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)
model.summary()

# -----------------------------
# 5. Phase 1: train the new top layers only
# -----------------------------
print("\n===== Phase 1: training top layers =====")
checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
    MODEL_FILE, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True, verbose=1
)

history1 = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
    class_weight=class_weight, callbacks=[checkpoint1, early_stop]
)
best_val_phase1 = max(history1.history["val_accuracy"])

# -----------------------------
# 6. Phase 2: fine-tune the last layers of MobileNetV2
# -----------------------------
print("\n===== Phase 2: fine-tuning =====")
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),   # very small learning rate for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=metrics,
)

checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
    MODEL_FILE, monitor="val_accuracy", save_best_only=True, mode="max",
    initial_value_threshold=best_val_phase1, verbose=1   # only overwrite if better than phase 1
)
early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True, verbose=1
)

model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
    class_weight=class_weight, callbacks=[checkpoint2, early_stop2]
)

# -----------------------------
# 7. Evaluate the best saved model on the test set
# -----------------------------
print("\nEvaluating best model on test dataset...")
best_model = tf.keras.models.load_model(MODEL_FILE)
test_loss, test_acc, test_top3 = best_model.evaluate(test_ds)

print("\n========== FINAL RESULTS ==========")
print(f"Test loss:           {test_loss:.4f}")
print(f"Test accuracy:       {test_acc * 100:.2f}%")
print(f"Test top-3 accuracy: {test_top3 * 100:.2f}%")
print(f"Saved model:         {MODEL_FILE}")
print("Saved class names:   model/class_names.json")