import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
import numpy as np
from tensorflow.keras.preprocessing import image


loaded_model = tf.keras.models.load_model('model/pothole_resnet50_v2.h5')

image_path = "uploads/download.jpg"
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x /= 255
xe = np.expand_dims(x, axis=0)
classs = loaded_model.predict(xe)[0,0]

print(classs)