import numpy as np
import tensorflow as tf
import os
import pickle
import random
import re
GCS_PREFIX = 'gs://uswow/records/'
IMG_WIDTH = 339
IMG_HEIGHT = 424
SHARD_SIZE = 1024
AUTO = tf.data.experimental.AUTOTUNE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/mrthinger/wow/google-key.json'
imgs_per_record = SHARD_SIZE


class_map = pickle.load(open('class_map.p', 'rb'))
num_classes = len(class_map)

tensor_map = tf.constant(class_map, dtype=tf.int64)   
tensor_num_classes = tf.constant(len(class_map))


def evaluate_dataset(dataset, show_imgs=False, wait_time=0.0):
    #evaluation dependancies
    import cv2
    import time
    last_time = time.time()

    for elem in dataset:
        time.sleep(wait_time)
        print('load took {} seconds'.format(time.time()-last_time-wait_time))

        if show_imgs:
            seqs = elem[0].numpy()
            for i, seq in enumerate(seqs):
                    
                for img in seq:
                    cv2.imshow(f'WoW SEQ replay', cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    time.sleep(.05)

            cv2.destroyAllWindows()
 
        last_time = time.time()

def read_tfrecord(example):

    assert tensor_map is not None, 'class_map was never set'
    assert tensor_num_classes is not None, 'tensor_num_classes never set'

    features = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "label": tf.io.FixedLenFeature([7], tf.int64)  
    }
    example = tf.io.parse_single_example(example, features)

    img = tf.image.decode_jpeg(example['image'], channels=3)
    # image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.bfloat16)
    img = tf.reshape(img, [IMG_WIDTH, IMG_HEIGHT, 3]) # explicit size will be needed for TPU

    label = example['label']
    #lookup [0, 0, 0, 0, 0, 0, 0] in dict
    label = tf.stack([label] * num_classes)
    row_matches = tf.reduce_all(tf.equal(tensor_map, label), axis=-1)
    ind = tf.where(row_matches)[0][0]
    #convert index to one-hot
    onehot = tf.one_hot(indices=ind, depth=tensor_num_classes)

    return img, onehot



def load_dataset(filenames, batch_size, rand_buffer, sequence_length, train_set):
    options = tf.data.Options()
    options.experimental_slack = True
    options.experimental_optimization.parallel_batch = True
    
    byte_buffer_size = 1e+7
    print('buffer size: '+str(byte_buffer_size/1e+9) + 'GB')

    dataset = tf.data.TFRecordDataset(filenames, buffer_size=int(byte_buffer_size))
    
    dataset = dataset.with_options(options)
    
    dataset = dataset.repeat()
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.window(sequence_length+1, shift=sequence_length//4, stride=2, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(sequence_length+1, drop_remainder=True), y.batch(sequence_length+1,  drop_remainder=True))))
    

    def shift(img, label):
        label = tf.roll(label, shift=-1, axis=0)
        label = tf.slice(label, [0,0], [sequence_length, 47])
        img = tf.slice(img, [0, 0, 0, 0], [sequence_length, IMG_WIDTH, IMG_HEIGHT , 3])
        return (img, label)

    dataset = dataset.map(shift, num_parallel_calls=AUTO)

    if train_set:
        dataset = dataset.shuffle(rand_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
   
    dataset = dataset.prefetch(AUTO)

    return dataset

def get_datasets(batch_size=4, val_split=.1, rand_buffer=425, sequence_length=10):
    filenames = tf.io.gfile.glob(GCS_PREFIX + '*.tfrec')

    # filenames = tf.io.gfile.glob('D:/wowrecords/' + '*.tfrec')
    # filenames.sort(key=lambda name: int(re.sub(r'\D', '', name)))
    # filenames.sort(key=lambda name: name.split('_')[2])

    split = int(len(filenames) * val_split)

    random.Random(1337).shuffle(filenames)

    training_filenames = filenames[split:]


    validation_filenames = filenames[:split]

    train_dataset = load_dataset(training_filenames, batch_size, rand_buffer, sequence_length, True)


    # evaluate_dataset(train_dataset, show_imgs=True)



    val_dataset = load_dataset(validation_filenames, batch_size, rand_buffer, sequence_length, False)

    

    
    # evaluate_dataset(train_dataset, show_imgs=True, wait_time=0.05)

    train_steps = ((imgs_per_record * len(training_filenames)) // batch_size)  // (sequence_length // 4)
    val_steps = ((imgs_per_record * len(validation_filenames)) // batch_size) // (sequence_length // 4)

    return train_dataset, val_dataset, train_steps, val_steps




if __name__ == "__main__":
    get_datasets()
