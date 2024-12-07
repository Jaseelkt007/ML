import os
import csv
from PIL import Image
from tqdm import tqdm

resized_folder = '/home/RUS_CIP/st186731/revized_images/test'
final_folder = '/home/RUS_CIP/st186731/revized_images/test/binary'
csv_file_path = '/home/RUS_CIP/st186731/dl-lab-24w-team04/labels/test.csv'

if not os.path.exists(final_folder):
    os.makedirs(final_folder)

# Create output folder structure for each class
for i in range(2):
    class_folder = os.path.join(final_folder, f'class_{i}')
    os.makedirs(class_folder, exist_ok=True)

with open(csv_file_path, encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)
    image_name_index = header.index("Image name")
    grade_index = header.index("label")

    # Loop through each row in CSV and process images based on class
    for row in tqdm(csv_reader, desc='Processing images', unit='image'):
        image_name = row[image_name_index]
        label = row[grade_index]

        # Load the preprocessed image
        img_path = os.path.join(resized_folder, f"{image_name}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} doesn't exist")
            continue  # Skip if the file doesn't exist

        image = Image.open(img_path)
        class_folder = os.path.join(final_folder, f'class_{label}')

        # Save the original image in the respective folder
        original_image_name = f"{image_name}.jpg"
        if not os.path.exists(os.path.join(class_folder, original_image_name)):  # Save only if it doesn't exist
            image.save(os.path.join(class_folder, original_image_name))