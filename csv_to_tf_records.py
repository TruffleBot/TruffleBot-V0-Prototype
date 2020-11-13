import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm, trange
import cv2





base_path = Path.home() / 'projects' / 'WOW'
imgs_dir = base_path / 'IMGS'
data_path = base_path  / 'data.csv'
records_path = base_path  / 'tfrecords'
GCS_PREFIX = 'gs://nnwow/m1_tfrecords/'
IMG_WIDTH = 339
IMG_HEIGHT = 424
SHARD_SIZE = 1024
AUTO = tf.data.experimental.AUTOTUNE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(base_path / 'google-key.json')


data = pd.read_csv(str(data_path), header=None)
X_names = data.iloc[:, 0]
y_labels = data.iloc[:, 1:]

num_samples = data.shape[0]



def process_img(file_path, label): 
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], preserve_aspect_ratio=False)

    #to jpg
    img = tf.cast(img, tf.uint8)
    img = tf.image.encode_jpeg(img, quality=100, optimize_size=True, chroma_downsampling=False)
   
   
    #convert to numpy
    # img = img.numpy()
    #print(img.shape)
    #show img filesize
    #print(img.nbytes)

    #replay video
    # cv2.imshow('WoW replay', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    return img, label


def play_img(encoded_jpeg): 
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(encoded_jpeg, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    #convert to numpy
    img = img.numpy()

    #replay video
    cv2.imshow('WoW replay', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return img


def load_data():
    Xs = []
    Ys = []
    print('LOADING DATA:')
    for i in tqdm(range(num_samples)):
        path = str(imgs_dir / X_names.iloc[i])

        if not os.path.exists(path):
            print(f'Skipped {path} because it did not exist')
            continue

        y_label = y_labels.iloc[i].to_numpy()

        Xs.append(path)
        Ys.append(y_label)
    
    imgs_paths = tf.data.Dataset.from_tensor_slices(Xs)
    labels = tf.data.Dataset.from_tensor_slices(Ys)


    return tf.data.Dataset.zip((imgs_paths, labels))
    
def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def to_tfrecord(img_bytes, label):  
  feature = {
      "image": _bytestring_feature([img_bytes]),
      "label": _int_feature(label),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))
  

if __name__ == '__main__':
    dataset_paths = load_data()
    dataset2_imgs = dataset_paths.map(process_img, num_parallel_calls=AUTO)


    print('MAKING/UPLOADING RECORDS:')
    for shard, (jpegs, labels) in enumerate(tqdm(dataset2_imgs.batch(SHARD_SIZE), desc='Total', total=int(num_samples/SHARD_SIZE))):
        np_jpegs = jpegs.numpy()
        np_labels = labels.numpy()

        outfile_name = GCS_PREFIX + f'wow_rec_{shard}.tfrec'
        batch_size = np_jpegs.shape[0]
        with tf.io.TFRecordWriter(outfile_name) as out_file:
            for i in trange(batch_size, desc='Current Shard', leave=False):
                example = to_tfrecord(np_jpegs[i], np_labels[i])
                out_file.write(example.SerializeToString())