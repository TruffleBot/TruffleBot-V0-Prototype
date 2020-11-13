#%% setup
import tensorflow as tf
import numpy as np
import datetime
import os
import pickle

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, TimeDistributed, LSTM, GRU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD

from load_gcs_data import IMG_WIDTH, IMG_HEIGHT, get_datasets, num_classes
from helpers import f1_m, precision_m, recall_m, get_weighted_loss, weighted_categorical_crossentropy

#loss: 1.3236 - accuracy: 0.8150 - f1_m: 0.7718 - val_loss: 2.5592 - val_accuracy: 0.6189 - val_f1_m: 0.5810
#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v3_weights/resnetv2_adamtest/weights.tf'

#loss: 0.6375 - accuracy: 0.9180 - f1_m: 0.9026 - val_loss: 3.1489 - val_accuracy: 0.5296 - val_f1_m: 0.5076
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v3_weights/resv2_sgd_test/weights.tf'

#loss: 0.4830 - accuracy: 0.9248 - f1_m: 0.9249 - val_loss: 2.4242 - val_accuracy: 0.7184 - val_f1_m: 0.7020
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v3_weights/m3_adam/weights.tf'

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = num_classes
BATCH_SIZE = 1
SEQUENCE_LENGTH = 1

setparam={
    'stateful':True,
    'train_cnn': True,
    'dropout':0.0,
    'initial_lr': .01,
    'epochs': 30
    }


print(f'Number of outputs: {num_classes}')

class_weights = pickle.load(open('weights.p', 'rb'))
class_weights = np.array(list(class_weights.values()))
print(f'Class weights: {class_weights}')


def make_model(params):

    #CNN DEFININITION
    cnn_model = keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights=None,#doesnt work
        input_shape=INPUT_SHAPE,
        pooling='avg')

    if not params['train_cnn']:
        for layer in cnn_model.layers:
            layer.trainable = False
    

    #COMBINED MODEL DEFINITION
    input_layer = keras.Input(shape=((SEQUENCE_LENGTH,)+INPUT_SHAPE), batch_size=BATCH_SIZE)
    cnn_layer = TimeDistributed(cnn_model, input_shape=(SEQUENCE_LENGTH,)+INPUT_SHAPE, name='cnn_timedist')(input_layer)

    #ADDED DROPOUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lstm_layer = GRU(1024, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'])(cnn_layer)

    predictions = Dense(NUM_OUTPUTS, activation = 'softmax', dtype=tf.float32)(lstm_layer)



    model = Model(inputs = input_layer, outputs = predictions)

    

    print('Compiling model...')

    opt = SGD(learning_rate=params['initial_lr'], nesterov=True, momentum=.9)
    loss = weighted_categorical_crossentropy(class_weights)

    model.compile(
        opt,
        loss=loss,
        #loss='categorical_crossentropy',
        #loss=get_weighted_loss(class_weights),
        metrics=['accuracy',f1_m])

    
    return model





def get_configured_model():
    print('Defining model...')
    model = make_model(setparam)
    print('Loading weights...')
    model.load_weights(WEIGHT_FILE)

    return model

