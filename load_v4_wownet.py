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

import tensorflow_addons as tfa  

from load_gcs_data import IMG_WIDTH, IMG_HEIGHT, get_datasets, num_classes
from helpers import f1_m, precision_m, recall_m, get_weighted_loss, weighted_categorical_crossentropy

#no idea what stats were here
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/7-lstm_resnetv2_AdamW_heavy_reg/weights.tf'
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/8_lstm_resnetv2_ADAM_lightreg/weights.tf'


#loss: 0.4934 - accuracy: 0.9344 - f1_m: 0.9335 - val_loss: 3.2590 - val_accuracy: 0.4578 - val_f1_m: 0.4207
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/7-lstm_resnetv2_AdamW_heavy_reg_10/weights.tf'

#loss: 0.3673 - accuracy: 0.9484 - f1_m: 0.9481 - val_loss: 2.4402 - val_accuracy: 0.7369 - val_f1_m: 0.7210
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/7-lstm_resnetv2_AdamW_heavy_reg_20/weights.tf'

#loss: 0.8257 - accuracy: 0.8864 - f1_m: 0.8825 - val_loss: 2.5175 - val_accuracy: 0.6041 - val_f1_m: 0.5647
# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/8_lstm_resnetv2_ADAM_lightreg_10/weights.tf'

#oss: 0.5787 - accuracy: 0.9182 - f1_m: 0.9174 - val_loss: 2.3850 -val_accuracy: 0.7172 - val_f1_m: 0.7048
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v4_weights/8_lstm_resnetv2_ADAM_lightreg_20/weights.tf'

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = num_classes
BATCH_SIZE = 1
SEQUENCE_LENGTH = 1

setparam={
    'stateful':True,
    'train_cnn': True,
    'dropout':0,
    'initial_lr': .001,
    'epochs': 5 * 2
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
    lstm_layer = LSTM(1024, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'])(cnn_layer)

    predictions = Dense(NUM_OUTPUTS, activation='softmax', dtype=tf.float32)(lstm_layer)

    model = Model(inputs = input_layer, outputs = predictions)

    

    print('Compiling model...')

    loss = weighted_categorical_crossentropy(class_weights)

    #opt = tfa.optimizers.extend_with_decoupled_weight_decay(SGD)(weight_decay=0.0001, learning_rate=['initial_lr'], nesterov=True, momentum=.9)
    opt = tfa.optimizers.AdamW(0.0001, learning_rate=params['initial_lr'])
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

