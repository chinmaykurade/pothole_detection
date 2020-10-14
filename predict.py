import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from keras.preprocessing import image


loaded_model = tf.keras.models.load_model('model/pothole')

image_path = "download.jpg"
img = image.load_img(image_path, target_size=(150, 150))
x = image.img_to_array(img)
xe = np.expand_dims(x, axis=0)
classs = loaded_model.predict(xe)[0,0]
