import numpy as np
import tensorflow as tf
import os
import pickle
import random
import re
from load_v8_wownet import get_configured_model
from tqdm import trange, tqdm
GCS_PREFIX = 'gs://uswow/records/'
IMG_WIDTH = 339
IMG_HEIGHT = 424
SHARD_SIZE = 1024
AUTO = tf.data.experimental.AUTOTUNE
imgs_per_record = 1024



def read_tfrecord(example):

    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "label": tf.io.FixedLenFeature([7], tf.int64)  
    }
    example = tf.io.parse_single_example(example, features)

    img = example['image']
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float16)
    # img = tf.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 3]) # explicit size will be needed for TPU

    label = example['label']

    return img, label



def load_dataset(filenames, batch_size):
    options = tf.data.Options()
    options.experimental_slack = True
    options.experimental_optimization.parallel_batch = True
    
    byte_buffer_size = 1e+7
    print('buffer size: '+str(byte_buffer_size/1e+9) + 'GB')

    dataset = tf.data.TFRecordDataset(filenames, buffer_size=int(byte_buffer_size))
    
    dataset = dataset.with_options(options)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True)
   
    dataset = dataset.prefetch(AUTO)

    return dataset

    
def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _floats_feature(list_of_floats): #can be float32 or float64
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def to_tfrecord(cnn_floats, label):  
  feature = {
      "image": _floats_feature(cnn_floats),
      "label": _int_feature(label),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))

def get_datasets(batch_size=1024):
    # filenames = tf.io.gfile.glob(GCS_PREFIX + '*.tfrec')

    filenames = tf.io.gfile.glob('D:/wowrecords/' + '*.tfrec')
    filenames.sort(key=lambda name: int(re.sub(r'\D', '', name)))
    filenames.sort(key=lambda name: name.split('_')[2])

    datasets = [load_dataset(filename, batch_size) for filename in filenames]

    # np_labels = np.empty((SHARD_SIZE*len(filenames), 7), dtype=np.int8)

    freq = np.zeros((7,2))

    for dataset, filename in zip(datasets, tqdm(filenames)):


        shard_name = filename.split('\\')[-1]

        for i, res in  enumerate(tqdm(dataset, total=SHARD_SIZE//batch_size, leave=False, desc='Calculating shard: ' + shard_name)):
          labels = res[1].numpy()
          for sampleI in range(len(labels)):
            label = labels[sampleI]
            for arrI in range(len(label)):
              freq[arrI][int(label[arrI])] = freq[arrI][int(label[arrI])] + 1

    
    print(freq)

    np.save('binary_freq.npy', freq)

    return dataset




if __name__ == "__main__":
    get_datasets()
