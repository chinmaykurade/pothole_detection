# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:12:44 2020

@author: chinm
"""
from flask import Flask,render_template,url_for,request,redirect,send_from_directory
import os
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                        # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
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