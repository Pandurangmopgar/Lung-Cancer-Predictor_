from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import numpy as np
import tensorflow as tf


app = Flask(__name__)
# model = joblib.load('E:/Lung_Cancer_CNN/Data/Data/model.joblib/')
model_directory_path = 'E:/Lung_Cancer_CNN/Data/Data/model.joblib/'
# from tensorflow.keras.models import load_model

# Replace this with the path to your model.joblib directory
# model_directory_path = 'path/to/your/model.joblib/'

model = tf.keras.models.load_model(model_directory_path)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Remove alpha channel if it exists
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    resized_image = cv2.resize(image, (224, 224))
    resized_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(resized_image)
    class_label=['adinocarcinoma','carcinoma','normal','squamous']
    return jsonify({'prediction': class_label[np.argmax(prediction)]})

if __name__ == '__main__':
    app.run(debug=True)
