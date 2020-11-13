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
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights/drop45/weights.tf'


#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7vec_100/weights.tf'
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_KLDIV_100/weights.tf'

#64 seq 16b (BEST)
#loss: 5.5047e-04 - accuracy: 0.5620 - f1_m: 0.9439 - val_loss: 0.0028 - val_accuracy: 0.5412 - val_f1_m: 0.9426
LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_50/weights.tf'

#2048 seq
#loss: 1.5860e-04 - accuracy: 0.5759 - f1_m: 0.9800 - val_loss: 0.0012 - val_accuracy: 0.5014 - val_f1_m: 0.9606
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_100/weights.tf'

#loss: 8.5968e-05 - b5_a: 0.9979 - b3_a: 0.9943 - f1_m: 0.9891 - val_loss: 6.4739e-04 - val_b5_a: 0.9958 - val_b3_a: 0.9901 - val_f1_m: 0.9800
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_150/weights.tf'

#.1 to 1 weighting (not good)
# loss: 5.3513e-04 - accuracy: 0.5963 - f1_m: 0.9820 - val_loss: 0.0032 - val_accuracy: 0.5696 - val_f1_m: 0.9756  
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weights_v1_focalloss/weights.tf'


##below are started from best 50 focal
#16 seq 128batch
# loss: 7.5179e-05 - b5_a:0.9986 - b3_a: 0.9960 - f1_m: 0.9913 - val_loss: 0.0067 - val_b5_a: 0.9946 - val_b3_a:0.9922 - val_f1_m: 0.9671
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_16seq_128batch/weights.tf'

#32 seq 128 batch (val data cut down to 8 files)
#loss: 6.6810e-05 - b5_a: 0.9986 - b3_a: 0.9958 - f1_m: 0.9914 - val_loss: 0.0144 - val_b5_a: 0.9859 - val_b3_a: 0.9815 - val_f1_m: 0.9340
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_32seq_128batch/weights.tf'


#32 seq 64 batch
#loss: 5.0557e-04 - b5_a: 0.9896 - b3_a: 0.9726 - f1_m: 0.9470 - val_loss: 0.0040 - val_b5_a: 0.9841 - val_b3_a: 0.9671 - val_f1_m: 0.9199
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_32seq_64batch/weights.tf'


#64seq 16 batch
##loss: 7.2053e-04 - b5_a: 0.9853 - b3_a: 0.9628 - f1_m: 0.9278 - val_loss: 0.0019 - val_b5_a: 0.9808 - val_b3_a: 0.9568 - val_f1_m: 0.9067
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_64seq_16batch/weights.tf'

#FROM SCRATCH
#64seq 64 batch
# loss: 2.4557e-04 - b5_a: 0.9947 - b3_a: 0.9857 - f1_m: 0.9719 - val_loss: 0.0051 - val_b5_a: 0.9893 - val_b3_a: 0.9808 - val_f1_m: 0.9447
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v8_7_weighted_focalloss_64seq_64batch/weights.tf'

# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm7/v9_64s_64b_4gamma/weights.tf'

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