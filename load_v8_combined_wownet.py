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


# WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v5_weights/v6_megaseq_predictFUTURE_STATEFUL_10/weights.tf'

#loss: 0.1049 - accuracy: 0.9759 - f1_m: 0.9760 - val_loss: 0.0544 - val_accuracy: 0.9891 - val_f1_m: 0.9892
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/10/weights.tf'

#loss: 0.0671 - accuracy: 0.9855 - f1_m: 0.9855 - val_loss: 0.0350 - val_accuracy: 0.9954 - val_f1_m: 0.9954
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/20/weights.tf'

#loss: 0.0328 - accuracy: 0.9932 - f1_m: 0.9932 - val_loss: 0.0212 - val_accuracy: 0.9982 - val_f1_m: 0.9981
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/30/weights.tf'

#loss: 0.0145 - accuracy: 0.9971 - f1_m: 0.9972 - val_loss: 0.0198 - val_accuracy: 0.9992 - val_f1_m: 0.9993
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/40/weights.tf'

#--smaller sequence length .4 dropout as opposed to .2
#loss: 0.1103 - accuracy:0.9801 - f1_m: 0.9802 - val_loss: 0.0769 - val_accuracy: 0.9885 - val_f1_m: 0.9884
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/1_20/weights.tf'

#added recurrent dropout .4 48 sequence length
#loss: 0.1311 - accuracy: 0.9748 - f1_m: 0.9748 - val_loss: 0.0914 - val_accuracy: 0.9886 - val_f1_m: 0.9887
# LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/2_20/weights.tf'



#NEW CNN
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights/drop45/weights.tf'

#NEW LSTMS ON NEW CNN

#loss: 0.4136 - accuracy: 0.9128 - f1_m: 0.9123 - val_loss: 0.3722 - val_accuracy: 0.9132 - val_f1_m: 0.9129 
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/2dropcnn_45/weights.tf'

#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/2drop_32seq_16batch/weights.tf'

#loss: 0.1382 - accuracy: 0.9638 - f1_m: 0.9638 - val_loss: 0.3150 - val_accuracy: 0.8805 - val_f1_m: 0.8798  
#LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/2drop_2048seq_16batch/weights.tf'

# loss: 0.0091 - accuracy: 0.9987 - f1_m: 0.9987 - val_loss: 0.0928 - val_accuracy: 0.9936 - val_f1_m: 0.9936
LSTM_FILE = 'C:/Users/miste/Desktop/NNWoW/v8_weights_lstm/2drop_2048seq_16batch_160/weights.tf'


INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = num_classes
BATCH_SIZE = 1
SEQUENCE_LENGTH = 1

setparam={
    'stateful':True,
    'train_cnn': True,
    'dropout':0,
    'initial_lr': .0001,
    'epochs': 1
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

    model = Model(inputs = input_layer, outputs = cnn_layer)




    print('Loading Main Model weights...')
    model.load_weights(WEIGHT_FILE).expect_partial()
    

    lstm_input_layer = keras.Input(shape=((SEQUENCE_LENGTH,2048)), batch_size=BATCH_SIZE)
    lstm_layer = LSTM(2048, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm1')(lstm_input_layer)
    lstm_layer2 = LSTM(768, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm2')(lstm_layer)
    lstm_layer3 = LSTM(384, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'], name='lstm3')(lstm_layer2)
    predictions = TimeDistributed(Dense(NUM_OUTPUTS, activation='softmax', dtype=tf.float32, name='predictions'))(lstm_layer3)

    sub_lstm_model = Model(inputs=lstm_input_layer, outputs=predictions)

    print('Loading LSTM SubModel Weights')
    sub_lstm_model.load_weights(LSTM_FILE)

    combined_output = sub_lstm_model(model.output)
    combined_model = Model(inputs = model.input, outputs=combined_output)
    
    print('Compiling model...')

    loss = weighted_categorical_crossentropy(class_weights)
    opt = Adam(learning_rate=params['initial_lr'])
    combined_model.compile(
        opt,
        loss=loss,
        # loss='categorical_crossentropy',
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