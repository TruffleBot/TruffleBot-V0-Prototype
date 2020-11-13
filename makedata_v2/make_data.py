import time
import random
import string
import os
import sys
import queue
import threading

import numpy as np
from absl import flags
from wow_recorder import WoWRecorder
import cv2

#array=[keystateW, keystateA, keystateS, keystateD, keystateTab, keystateSpace, keystate1, keystate2]
IMG_TIME = .25
SUBDIVIDES = 5
SUBDIVIDE_TIME = IMG_TIME/SUBDIVIDES

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'output_dir',
    default=None,
    help='The directory where output data is stored. EX: C:/wowdata')



def randomString(stringLength=5):
    #Generate a random string of fixed length
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def make_filesys(output_dir):
    if not os.path.exists(output_dir + '/IMGS'):
        os.makedirs(output_dir + '/IMGS')

def write_buffer(buffer):
    with open(FLAGS.output_dir + '/data.csv', 'a') as fd:
        fd.write(buffer)

def image_writer(img_queue):
    while(True):
        if img_queue.not_empty:
            path, img = img_queue.get()
            cv2.imwrite(path, img)

def sample_writer(sample_q):
    buffer_samples = 0
    buffer = ''
    while(True):
        if sample_q.not_empty:
            img_path, key_list = sample_q.get()
            buffer = buffer + img_path + ','
            for key in key_list:
                buffer = buffer + str(key) + ','
            buffer = buffer[:-1]
            buffer = buffer + '\n'
            buffer_samples += 1
        if buffer_samples % 32:
            write_buffer(buffer)
            buffer_samples = 0
            buffer = ''
            




def screen_record(output_dir): 
    
    recorder = WoWRecorder()

    make_filesys(output_dir)
    img_dir = output_dir + '/IMGS'

    img_queue = queue.Queue()
    img_thread = threading.Thread(target=image_writer, args=(img_queue,), daemon=True)
    img_thread.start()

    indexing_queue = queue.Queue()
    sample_thread = threading.Thread(target=sample_writer, args=(indexing_queue,), daemon=True)
    sample_thread.start()

    i = 0
    name = randomString(5)
    
    print('Now collecting data')

    last_time = time.time()

    recording = True

    while(recording):

        printscreen = recorder.get_frame()

        img_name = name + str(i) +'.jpg'
        img_path = img_dir + '/' + img_name

        img_queue.put((img_path, printscreen))

        key_list = np.empty((5*8,), dtype=np.int8)

        last_key_time = time.time()
        for timestep in range(SUBDIVIDES):

            keys = recorder.get_keys()

            if keys == [1,1,1,1,0,0,1]:
                print('stopping data collection')
                recording = False

            for key_num, val in enumerate(keys):
                key_list[(key_num*SUBDIVIDES) + timestep] = val

            time_to_wait = SUBDIVIDE_TIME - (time.time() - last_key_time)
            if time_to_wait > 0:
                print('waiting: ' + str(time_to_wait))
                time.sleep(time_to_wait)

            last_key_time = time.time()

        
        indexing_queue.put((img_name, key_list))
            




        
        i+=1
        loop_time = time.time()-last_time
        print('loop took {} seconds'.format(loop_time))

        if IMG_TIME > loop_time:
            sleep_time = IMG_TIME - loop_time
            print('sleeping for {} seconds because we have extra time'.format(sleep_time))
            time.sleep(sleep_time)

        last_time = time.time()





def main():
    assert FLAGS.output_dir is not None, ('Provide output data path via --output_dir.')



    # while not recorder.get_wow_running():
    #     print('Waiting for WoW to open...')
    #     time.sleep(1)

    for i in range(4):
        print(f'starting in: {4-i}')
        time.sleep(1)




    screen_record(FLAGS.output_dir)
    
if __name__ == "__main__":
    FLAGS(sys.argv)
    main()