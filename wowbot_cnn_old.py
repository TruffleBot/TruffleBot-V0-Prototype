import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import cv2
from directkeys import PressKey, ReleaseKey, W, A, S, D, ONE, TAB, SPACE
from wow_recorder import WoWRecorder
import os
import pickle
import time

IMG_WIDTH = 339
IMG_HEIGHT = 424
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
model_names = ['wownet_inceptionresv2.h5', 'wownet_bigdata_ires2.h5', 'wownet_ires_onehot_1.h5',  'wownet_ires_onehot_2.h5', 'wownet_vgg16_onehot_1.h5', 'wownet_ires_onehot_4.h5']
selected_model_name = model_names[-1]

keys=[W, A, S, D, ONE, TAB, SPACE]

def press_keys(prediction):
    for key in keys:
        ReleaseKey(key)
    for key, pred in zip(keys, prediction):
        if(pred==1):
            PressKey(key)


def read_img(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

def process_img(img): 

    cv2.imwrite('temp.jpg', img)
    img = tf.io.read_file('temp.jpg')
    os.remove('temp.jpg')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=False)
    img = tf.cast(img, tf.uint8)
    img = tf.image.encode_jpeg(img, quality=100, optimize_size=True, chroma_downsampling=False)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 3])


    #replay video
    cv2.imshow('WoW replay', cv2.cvtColor(img.numpy()[0], cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return img

def fast_pro_img(img):

    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=False)
    img = tf.cast(img, tf.uint8)
    img = tf.image.encode_jpeg(img, quality=95, optimize_size=True, chroma_downsampling=False)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    channels = tf.unstack (img, axis=-1)
    img = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    img = tf.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 3])

    return img

def fastest_pro_img(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=False)
    channels = tf.unstack (img, axis=-1)
    img = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    img = tf.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 3])
    return img



model = load_model(selected_model_name)
print(model.summary())
recorder = WoWRecorder()
class_map = pickle.load(open('class_map.p', 'rb'))
last_time = time.time()

while(True):

    #800x600 windowed mode
    printscreen = recorder.get_frame()
    pred = model.predict(process_img(printscreen))[0]
    
    # print(pred)

    ind = np.argmax(pred)

    pred_keys = class_map[ind]

    os.system('cls')
    print(f'{pred_keys}')
    print('attackval: {0:.2f}\nstopval: {1:.2f}\nforwardval: {2:.2f}'.format(float(pred[8]), float(pred[0]), float(pred[3])))

    press_keys(pred_keys)

    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()