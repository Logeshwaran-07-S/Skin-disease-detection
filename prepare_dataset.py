import tensorflow as tf

# -----------------------------
# 1. Settings
# -----------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_PATH = "Dataset/Train"
TEST_PATH = "Dataset/Test"


# -----------------------------
# 2. Load training dataset
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    subset="training",
    seed=123
)


# -----------------------------
# 3. Load validation dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    subset="validation",
    seed=123
)


# -----------------------------
# 4. Load test dataset
# -----------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# 5. Display class information
# -----------------------------

class_names = train_dataset.class_names

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Number of classes:", len(class_names))

print("\nClass names:")

for number, name in enumerate(class_names):
    print(number, "=", name)


# -----------------------------
# 6. Display dataset sizes
# -----------------------------

print("\n==============================")
print("DATASET BATCH INFORMATION")
print("==============================")

print("Training batches:", tf.data.experimental.cardinality(train_dataset).numpy())

print("Validation batches:", tf.data.experimental.cardinality(validation_dataset).numpy())

print("Test batches:", tf.data.experimental.cardinality(test_dataset).numpy())


# -----------------------------
# 7. Normalize pixel values
# -----------------------------

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)


print("\nDataset preparation completed successfully!")