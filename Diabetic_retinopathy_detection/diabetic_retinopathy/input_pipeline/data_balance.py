# Minor augmentation to balance the classes
import os
import csv
import tensorflow as tf
from PIL import Image
from tqdm import tqdm



# Define augmentation functions for minor augmentation
def augment_image(image):  # Minor augmentation to balance the dataset
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02),  # Small rotation
        tf.keras.layers.RandomBrightness(0.05),
        tf.keras.layers.RandomFlip("horizontal"),
    ])
    return data_augmentation(image)

# Paths and folders
preprocessed_folder = '/home/RUS_CIP/st186731/revized_images/train'
augmented_folder = '/home/RUS_CIP/st186731/augmented_images/train'
csv_file_path = '/home/RUS_CIP/st186731/dl-lab-24w-team04/labels/train.csv'

if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

# Create output folder structure for each class
for i in range(2):
    class_folder = os.path.join(augmented_folder, f'class_{i}')
    os.makedirs(class_folder, exist_ok=True)

# Set target augmentation limits (total count, including originals)
target_augmentation = {'0': 300, '1': 50}  # Total images needed per class
class_counters = {'0': 0, '1': 0}  # Track the count of images per class (original + augmented)
def balance_dataset():
    # Read the CSV file and process each image
    with open(csv_file_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        image_name_index = header.index("Image name")
        grade_index = header.index("label")

        # Loop through each row in CSV and process images based on class
        for row in tqdm(csv_reader, desc='Processing images', unit='image'):
            image_name = row[image_name_index]
            label = row[grade_index]
            label_str = str(label)

            # Load the preprocessed image
            img_path = os.path.join(preprocessed_folder, f"{image_name}.jpg")
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} doesn't exist")
                continue  # Skip if the file doesn't exist

            image = Image.open(img_path)
            image_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert PIL image to numpy array
            image_array = tf.image.convert_image_dtype(image_array, dtype=tf.float32)  # Scale pixel values to [0,1]

            # Define where to save images based on class label
            class_folder = os.path.join(augmented_folder, f'class_{label}')

            # Save the original image as `aug_0` for each file
            original_image_name = f"{image_name}_aug_0.jpg"
            if not os.path.exists(os.path.join(class_folder, original_image_name)):  # Save only if it doesn't exist
                image.save(os.path.join(class_folder, original_image_name))
                class_counters[label_str] += 1  # Count the augmented image

            # Generate one augmented image as `aug_1` only if the target count is not yet met
            if class_counters[label_str] < target_augmentation[label_str]:
                augmented_image = augment_image(image_array)
                augmented_image = tf.keras.preprocessing.image.array_to_img(augmented_image)  # Convert back to PIL image
                aug_image_name = f"{image_name}_aug_1.jpg"
                augmented_image.save(os.path.join(class_folder, aug_image_name))
                class_counters[label_str] += 1  # Count the augmented image



if __name__ == '__main__':
    balance_dataset()
    print("Original and augmented images saved with consistent naming.")