from tensorflow.keras.models import load_model
from flask import Flask, flash, request, app, jsonify, url_for,render_template
import numpy as np
import cv2
import imghdr
import tensorflow as tf

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpeg','jpg','png','bmp'])

app=Flask(__name__)
model=load_model('happy_sad_imageclassificationmodel.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    #data = request.form.values()
    nparr = np.fromstring(request.data, np.uint8)
    #decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_resized = tf.image.resize(img, (256,256))
    y_pred = model.predict(np.expand_dims(img_resized/255,0))
    if y_pred < 0.5:
        return jsonify('Predicted class is Happy!')
    else: 
        return jsonify('Predicted class is Sad')
    
@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files["image"]
    nparr = np.fromstring(uploaded_file.read(), np.uint8)
    #data = request.form.values()
    #nparr = np.fromstring(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #return render_template("home.html", prediction_text= '{}'.format(img.dtypes()))
  
    img_resized = tf.image.resize(img, (256,256))
    y_pred = model.predict(np.expand_dims(img_resized/255,0))
    if y_pred < 0.5:
        return render_template("home.html", prediction_text='Predicted class is Happy!')
    else: 
        return render_template("home.html", prediction_text='Predicted class is Sad!')


if __name__=="__main__":
    app.run(debug=True)

