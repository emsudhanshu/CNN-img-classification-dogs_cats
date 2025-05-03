from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

#loading the model for image label testing
model = load_model('./cat_dog_model.h5')

def predict_image(model, img_path):
    # processing the image to make it ready to feed into the model for prediction
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    prediction = model.predict(img_array)[0][0]  # get prediction value

    if prediction > 0.5:
        print(f"Prediction: dog ({prediction:.2f})")
    else:
        print(f"Prediction: Cat ({1 - prediction:.2f})")

imgPath = "./cat.14.jpg"  # input image 1
predict_image(model, imgPath);

imgPath = "./dog.5.jpg" # image image 2
predict_image(model, imgPath);

