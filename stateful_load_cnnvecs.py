import numpy as np
import tensorflow as tf
import os
import pickle
import random
import cv2
from record_splitter import get_filelists
#SHUFFLE IS OFF ATM
IMG_WIDTH = 339
IMG_HEIGHT = 424
SHARD_SIZE = 1024
AUTO = tf.data.experimental.AUTOTUNE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/root/WoW/google-key.json'
imgs_per_record = SHARD_SIZE


class_map = pickle.load(open('class_map.p', 'rb'))
num_classes = len(class_map)

tensor_map = tf.constant(class_map, dtype=tf.int64)   
tensor_num_classes = tf.constant(len(class_map))

def read_tfrecord(example):

    assert tensor_map is not None, 'class_map was never set'
    assert tensor_num_classes is not None, 'tensor_num_classes never set'

    features = {
        "image": tf.io.FixedLenFeature([2048], tf.float32), 
        "label": tf.io.FixedLenFeature([7], tf.int64)  
    }
    example = tf.io.parse_single_example(example, features)

    img = example['image']

    label = example['label']
    #lookup [0, 0, 0, 0, 0, 0, 0] in dict
    label = tf.stack([label] * num_classes)
    row_matches = tf.reduce_all(tf.equal(tensor_map, label), axis=-1)
    ind = tf.where(row_matches)[0][0]
    #convert index to one-hot
    onehot = tf.one_hot(indices=ind, depth=tensor_num_classes)

    return img, onehot

def load_dataset(filenames, sequence_length):

    options = tf.data.Options()
    options.experimental_slack = True
    options.experimental_optimization.parallel_batch = True
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=None)
    dataset = dataset.with_options(options)
    dataset = dataset.repeat()
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)


    dataset = dataset.window(sequence_length+1, shift=sequence_length*2, stride=2, drop_remainder=True)
    dataset = dataset.flat_map(lambda x,y : tf.data.Dataset.zip((x.batch(sequence_length+1, drop_remainder=True), y.batch(sequence_length+1,  drop_remainder=True))))
    

    def shift(img, label):
        label = tf.roll(label, shift=-1, axis=0)
        label = tf.slice(label, [0,0], [sequence_length, 47])
        img = tf.slice(img, [0, 0, 0, 0], [sequence_length, IMG_WIDTH, IMG_HEIGHT , 3])
        return (img, label)

    dataset = dataset.map(shift, num_parallel_calls=AUTO)


    return dataset

def combined_dataset(dataset, length_of_tensorslices):


    dataset = dataset.interleave(lambda x: x.batch(length_of_tensorslices, drop_remainder=True),num_parallel_calls=AUTO )
    dataset = dataset.prefetch(AUTO)

    return dataset

def get_datasets(sequence_length=10):

    train_filename_set, val_filename_set = get_filelists() 


    train_sequences = [load_dataset(filenames, sequence_length) for filenames in train_filename_set]

    train_data = tf.data.Dataset.from_tensor_slices(train_sequences)
    train_data = combined_dataset(train_data, len(train_sequences))
    
    val_sequences = [load_dataset(filenames, sequence_length) for filenames in val_filename_set]

    val_data = tf.data.Dataset.from_tensor_slices(val_sequences)
    val_data = combined_dataset(val_data, len(val_sequences))


    
    

    train_steps = (imgs_per_record * 82) // sequence_length
    val_steps = (imgs_per_record * 3) // sequence_length

    return train_data, val_data, train_steps, val_steps




if __name__ == "__main__":
    get_datasets()
