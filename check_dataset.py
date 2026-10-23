import os

train_path = "Dataset/Train"
test_path = "Dataset/Test"


def count_images(folder_path):
    total = 0

    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)

        if os.path.isdir(class_path):
            images = [
                file for file in os.listdir(class_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"{class_name}: {len(images)} images")
            total += len(images)

    print(f"Total images: {total}")


print("========== TRAIN DATASET ==========")
count_images(train_path)

print("\n========== TEST DATASET ==========")
count_images(test_path)