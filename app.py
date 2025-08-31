from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model once when the app starts
model = tf.keras.models.load_model('model3.hdf5', compile=False)

# Classes
classes = {0: 'Normal Tire', 1: 'Cracked Tire'}
THRESHOLD = 0.5

def preprocess_image_from_bytes(image_bytes: bytes):
    image_stream = io.BytesIO(image_bytes)
    img = Image.open(image_stream).convert('L')
    img = img.resize((300, 300), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # channel
    img_array = np.expand_dims(img_array, axis=0)   # batch
    return img_array

def predict_image_from_bytes(image_bytes: bytes) -> str:
    preprocessed = preprocess_image_from_bytes(image_bytes)
    prediction = model.predict(preprocessed)
    class_index = 1 if prediction[0][0] >= THRESHOLD else 0
    return classes[class_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file selected", image_url=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected", image_url=None)
        
        image_bytes = file.read()
        if not image_bytes:
            return render_template('index.html', prediction="Empty file uploaded", image_url=None)

        prediction = predict_image_from_bytes(image_bytes)

        mime_type = file.mimetype or 'image/jpeg'
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{b64_image}"

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
