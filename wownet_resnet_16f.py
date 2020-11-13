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

# tf.compat.v1.disable_eager_execution()
#tf.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

from stateful_load_gcs_data import IMG_WIDTH, IMG_HEIGHT, get_datasets, num_classes
from helpers import f1_m, precision_m, recall_m, get_weighted_loss, weighted_categorical_crossentropy



##CONFIG
USE_TPU = False 
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
NUM_OUTPUTS = num_classes
SEQUENCE_LENGTH = 32
PER_CORE_BATCH_SIZE = 8
WEIGHT_FILE = '/root/WoW/starting_weights/weights.tf' 
SAVE_FILE = 'gs://nnwow/lstm_models/resnet_bf16_40_stateful/weights.tf'


setparam={
    'activation':'relu', #elu
    'conv_model':'resnet',#'iresnet2'
    'stateful':True,
    'train_cnn': False,
    'dropout':0.2,
    'optimizer':Adam,
    'lr': .000033,
    'epochs': 30
    }


print(f'Number of outputs: {num_classes}')

class_weights = pickle.load(open('weights.p', 'rb'))
class_weights = np.array(list(class_weights.values()))
print(f'Class weights: {class_weights}')


if USE_TPU:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='instance-1', project='openset-ml', zone='europe-west4-a')  
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    BATCH_SIZE = PER_CORE_BATCH_SIZE * strategy.num_replicas_in_sync
else:
    BATCH_SIZE = PER_CORE_BATCH_SIZE




train_dataset, val_dataset, train_steps, val_steps = get_datasets(sequence_length=SEQUENCE_LENGTH)


def batch(params):
    if USE_TPU:
        with strategy.scope():
            model = make_model(params)
    else:
        model = make_model(params)

    log_dir="gs://nnwow/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    callbacks_list = []#tensorboard_callback]

    out = model.fit(train_dataset,
                    steps_per_epoch=train_steps,
                    validation_data=val_dataset,
                    validation_steps=val_steps,
                    epochs=params['epochs'],
                    verbose=1,
                    # validation_freq=[params['epochs']],
                    callbacks=callbacks_list
    )

    return out, model




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

    #ADDED DROPOUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lstm_layer = GRU(1024, return_sequences=True, stateful=params['stateful'], dropout=params['dropout'])(cnn_layer)

    predictions = Dense(NUM_OUTPUTS, activation = 'softmax', dtype=tf.float32)(lstm_layer)



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

    print('loading weights')
    model.load_weights(WEIGHT_FILE)
    
    return model




_, model = batch(setparam)

print('saving weights')
model.save_weights(SAVE_FILE)
