#importing the libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import zipfile
import os
import shutil
import numpy as np
from PIL import UnidentifiedImageError
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# extracting the zip file i.e. train.zip

train_zip_path = '../train.zip' #file path of train.zip in my local system

#below function will extract the images
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

#specifying the path for extracted train images
train_extract_path = './train_extracted'
os.makedirs(train_extract_path, exist_ok=True)
base_dir = extract_and_fix_structure(train_zip_path, train_extract_path)
base_dir = os.path.join(base_dir, 'train')


#organizing the data from in respective cats and dogs folders
#making the folders her
os.makedirs(os.path.join(base_dir, 'cats'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'dogs'), exist_ok=True)

#so basically identifying the labels from the file name like if filename contains cats then moving it in cat folder and same goes for dog as wel
for filename in os.listdir(base_dir):
    file_path = os.path.join(base_dir, filename)
    if os.path.isfile(file_path):
        if 'cat' in filename.lower():
            shutil.move(file_path, os.path.join(base_dir, 'cats', filename))
        elif 'dog' in filename.lower():
            shutil.move(file_path, os.path.join(base_dir, 'dogs', filename))

print("images moved into 'cats/' and 'dogs/'")

#  here i am removing the currupt images because when I extracted train.zip in macos some junk images also came

def clean_corrupt_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            try:
                load_img(os.path.join(root, f))
            except:
                os.remove(os.path.join(root, f))

clean_corrupt_images(base_dir)


# initializing the image generators each for training and validation set

#using 20% of train images for validation and rest 80% for training

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# building the model here
#as per the approach 3 I am using 3 convolution layers and respective maxpooling layers

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)), #1st con layer
    MaxPooling2D(2,2), 
    Conv2D(64, (3,3), activation='relu'), #2nd conv layer
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'), #3rd conv layer
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

#i m training the model

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10 #epochs are 10 here in the final approach
)

#plotting the accuracy vs epchos graph

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#plotting the loss vs epchos graph

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#saving the model

model.save('./cat_dog_model.h5')



# from here I am preparing the confusion matrix and classification report

# loading the previously stored model
model = load_model('cat_dog_model.h5') 

# generating the confusion matrix
# i am using validation data set to to make the confusion matrix as it labelled 
validation_generator.reset()
predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.tight_layout()
plt.show()

#printing the classification report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

