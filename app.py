# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:12:44 2020

@author: chinm
"""
from flask import Flask,render_template,url_for,request,redirect,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

loaded_model = tf.keras.models.load_model('model/pothole_resnet50_v2.h5')

def predict_image(image_path,model):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    xe = np.expand_dims(x, axis=0)
    pred = model.predict(xe)
    classs = pred[0,0]
    return classs

if not 'uploads' in os.listdir('.'):
    os.mkdir('uploads')

# app = Flask(__name__) # to make the app run without any
app = Flask(__name__)


@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        img = request.files['img']
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(
            basepath, 'uploads', secure_filename(img.filename))
        img.save(image_path)
        filename = secure_filename(img.filename)
        prediction = predict_image(image_path,loaded_model)
        return render_template('index.html',prediction=prediction,image_name=filename)

    else:
        return render_template('index.html')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    app.run(debug=True)