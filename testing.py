import shutil
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import zipfile

#loadng the trained model
model = load_model('./cat_Dog_model.h5')

# Paths of the train.zip extraction path
test_zip_path = './test1.zip'
test_extract_path = './test1/test1'
os.makedirs(test_extract_path, exist_ok=True)

# Function to extract zip and fix folder nesting
def extract_and_fix_structure(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    while True:
        files = os.listdir(extract_path)
        if len(files) == 1 and os.path.isdir(os.path.join(extract_path, files[0])):
            extract_path = os.path.join(extract_path, files[0])
        else:
            break
    return extract_path

# Extract and fix folder structure as per training and validation file same logic
test_dir = extract_and_fix_structure(test_zip_path, test_extract_path)

# Remove __MACOSX or junk folders
mac_folder = os.path.join(test_dir, "__MACOSX")
if os.path.exists(mac_folder):
    shutil.rmtree(mac_folder)

#Loading test images
test_images = []
valid_filenames = []  # Only keep filenames of valid images
test_filenames = sorted(os.listdir(test_dir))

for img_name in test_filenames:
    img_path = os.path.join(test_dir, img_name)
    if os.path.isdir(img_path):
        continue
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        test_images.append(img_array)
        valid_filenames.append(img_name)
    except Exception as e:
        print(f"Skipping invalid image {img_name}: {e}")

if len(test_images) == 0:
    raise ValueError("❌ No valid test images found!")

# using the strored trained model to predict the label of the testing images
test_images = np.array(test_images)
predictions = model.predict(test_images)
predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

# Creating submission.csv with id and labels
submission = pd.DataFrame({
    "id": [os.path.splitext(name)[0] for name in valid_filenames],
    "label": predicted_labels
})

#storing the file
submission.to_csv("submission.csv", index=False)
print("predictions saved to submission.csv!")



