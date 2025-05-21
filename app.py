from flask import Flask, request
from flask_cors import CORS

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
model = load_model('./cat_dog_model.h5')

def predict_image(model, image_file):
    # Load the image from bytes using PIL
    img = Image.open(image_file)
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]  # Output is a probability

    if prediction > 0.5:
        print(f"Prediction: Dog ({prediction:.2f})")
        return 'dog'
    else:
        print(f"Prediction: Cat ({1 - prediction:.2f})")
        return 'cat'

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Cat-Dog Classifier API Running'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image part', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image', 400

    result = predict_image(model, image_file)
    return result

# Run the Flask app on port 5001
app.run(debug=True, port=5001)
