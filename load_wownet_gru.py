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

#loss: 1.323 - accuracy: 0.8211 - f1_m: 0.8013 - val_loss: 1.35 - val_accuracy: 0.8219 - val_f1_m: 0.7846
#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_bf16_20/weights.tf'

#loss: 0.9531 - accuracy: 0.8460 - f1_m: 0.8403 - val_loss: 1.7182 - val_accuracy: 0.6877 - val_f1_m: 0.6895
#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_bf16_30/weights.tf'

##REPORT  HERE, resnet and gru were chosen for training speed bfloat16 on tpu, 1 second pattern recognition max, large batch sizes, shuffle windows
#loss: 0.1194 - accuracy: 0.9785 - f1_m: 0.9785 - val_loss: 3.4261 - val_accuracy: 0.6914 - val_f1_m: 0.6914
#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_bf16_40/weights.tf'

#WEIGHT_FILE = r'C:\Users\miste\Desktop\NNWoW\gru_weights\resnet_bf16_overfitTEST_80\weights.tf'
#WEIGHT_FILE = r'C:\Users\miste\Desktop\NNWoW\gru_weights\resnet_bf16_overfitTEST_190\weights.tf'

#loss: 0.6738 - accuracy: 0.8801 - f1_m: 0.8792 - val_loss: 0.9060 - val_accuracy: 0.8300 - val_f1_m: 0.8303
#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_stateful_10/weights.tf'

##REPORT
#loss: 0.6196 - accuracy: 0.8863 - f1_m: 0.8858 - val_loss: 0.6239 - val_accuracy: 0.8451 - val_f1_m: 0.8445
WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_stateful_30/weights.tf'


#WEIGHT_FILE = 'C:/Users/miste/Desktop/NNWoW/gru_weights/resnet_stateful_unwt_5/weights.tf'



INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = num_classes
BATCH_SIZE = 1
SEQUENCE_LENGTH = 1

setparam={
    'activation':'relu', #elu
    'conv_model':'resnet',#'iresnet2'
    'stateful':True,
    'train_cnn': True,
    'dropout':0,
    'optimizer':Adam,
    'lr': .00033,
    'epochs': 10
    }


print(f'Number of outputs: {num_classes}')

class_weights = pickle.load(open('weights.p', 'rb'))
class_weights = np.array(list(class_weights.values()))
print(f'Class weights: {class_weights}')


def make_model(params):

    #CNN DEFININITION
    conv_model_name = params['conv_model']
    conv_model = None

    if conv_model_name == 'vgg16':
        conv_model = keras.applications.vgg16.VGG16(
            include_top=False,
            weights=None,
            input_shape=INPUT_SHAPE,
            pooling='avg') #none vs avg???
    elif conv_model_name == 'iresnet2':
        conv_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights=None,#doesnt work
            input_shape=INPUT_SHAPE,
            pooling='avg')
    elif conv_model_name == 'dense':
        conv_model = keras.applications.densenet.DenseNet121(
            include_top=False,
            weights=None,#doesnt work
            input_shape=INPUT_SHAPE,
            pooling='avg')
    elif conv_model_name == 'resnet':
        conv_model = keras.applications.resnet.ResNet50(
            include_top=False,
            weights=None,#doesnt work
            input_shape=INPUT_SHAPE,
            pooling='avg')

    assert conv_model is not None, 'invalid conv_model name'
    if not params['train_cnn']:
        for layer in conv_model.layers:
            layer.trainable = False


    #flatten
    # flatten = Flatten()(conv_model.output)
    # cnn_model = Model(inputs = conv_model.input, outputs = flatten)

    cnn_model = conv_model

    #COMBINED MODEL DEFINITION
    input_layer = keras.Input(shape=((SEQUENCE_LENGTH,)+INPUT_SHAPE), batch_size=BATCH_SIZE)
    cnn_layer = TimeDistributed(cnn_model, input_shape=(SEQUENCE_LENGTH,)+INPUT_SHAPE, name='cnn_timedist')(input_layer)

    lstm_layer = GRU(1024, return_sequences=True, stateful=params['stateful'])(cnn_layer)

    predictions = Dense(NUM_OUTPUTS, activation = 'softmax')(lstm_layer)



    model = Model(inputs = input_layer, outputs = predictions)

    loss = weighted_categorical_crossentropy(class_weights)    

    print('Compiling model...')

    model.compile(
        params['optimizer'](params['lr']),
        loss=loss,
        #loss='categorical_crossentropy',
        #loss=get_weighted_loss(class_weights),
        metrics=['accuracy', f1_m])

    print(model.summary())
    
    return model





def get_configured_model():
    print('Defining model...')
    model = make_model(setparam)
    print('Loading weights...')
    model.load_weights(WEIGHT_FILE) #model.save_weights('gs://nnwow/lstm_models/weights_lstm_highbatch_lowseq_1/weights.tf')

    return model

