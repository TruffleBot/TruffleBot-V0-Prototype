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



#NEW CNN
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v9_weights/longseq_45/weights.tf'


#loss: 1.3159e-04 - b5_a: 0.9892 - b3_a: 0.9547 - f1_m: 0.9448 - val_loss: 0.0037 - val_b5_a: 0.9823 - val_b3_a: 0.9553 - val_f1_m: 0.9113
LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v9_weights_lstm/v9_64s_64b_4gamma_newcnn/weights.tf'

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = 7
BATCH_SIZE = 1
SEQUENCE_LENGTH = 1

setparam={
    'stateful':True,
    'train_cnn': True,
    'dropout':0,
    'initial_lr': .0001,
    'epochs': 1
    }


print(f'Number of outputs: {NUM_OUTPUTS}')

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

    model = Model(inputs = input_layer, outputs = cnn_layer)




    print('Loading Main Model weights...')
    model.load_weights(WEIGHT_FILE).expect_partial()
    

    lstm_input_layer = keras.Input(shape=((SEQUENCE_LENGTH,2048)), batch_size=BATCH_SIZE)
    lstm_layer = LSTM(2048, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm1')(lstm_input_layer)
    lstm_layer2 = LSTM(768, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm2')(lstm_layer)
    lstm_layer3 = LSTM(384, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm3')(lstm_layer2)
    predictions = TimeDistributed(Dense(NUM_OUTPUTS, activation='sigmoid', dtype=tf.float32, name='predictions'))(lstm_layer3)

    sub_lstm_model = Model(inputs=lstm_input_layer, outputs=predictions)

    print('Loading LSTM SubModel Weights')
    sub_lstm_model.load_weights(LSTM_FILE)

    combined_output = sub_lstm_model(model.output)
    combined_model = Model(inputs = model.input, outputs=combined_output)
    
    print('Compiling model...')

    # loss = weighted_categorical_crossentropy(class_weights)
    opt = Adam(learning_rate=params['initial_lr'])
    combined_model.compile(
        opt,
        # loss=loss,
        loss='binary_crossentropy',
        #loss=get_weighted_loss(class_weights),
        metrics=['accuracy',f1_m])

    
    return combined_model





def get_configured_model():
    print('Defining model...')
    model = make_model(setparam)
    # print(model.summary())

    return model

if __name__ == "__main__":
    get_configured_model()