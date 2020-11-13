import time
import random
import string
import os
import sys

from absl import flags
from wow_recorder import WoWRecorder
import cv2

#array=[keystateW, keystateA, keystateS, keystateD, keystate1, keystateTab, keystateSpace]

MAX_FPS = 24
rate = 1. / MAX_FPS

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

def write_buffer(buffer, output_dir):
    with open(output_dir + '/data.csv', 'a') as fd:
        fd.write(buffer)

def screen_record(output_dir): 
    
    recorder = WoWRecorder()

    make_filesys(output_dir)
    img_dir = output_dir + '/IMGS'


    i = 0
    last_time = time.time()
    name = randomString(5)
    
    sample_buffer = ''
    print('Now collecting data')
    while(True):

        printscreen = recorder.get_frame()
        keys = recorder.get_keys()

        if keys == [1,1,1,1,0,0,1]:
            print('stopping data collection')
            break

        img_name = name + str(i) +'.jpg'
        img_path = img_dir + '/' + img_name

        cv2.imwrite(img_path, printscreen)

        sample = img_name + "," +str(keys[0]) +',' + str(keys[1]) +',' +str(keys[2]) +',' +str(keys[3]) +',' +str(keys[4]) +',' +str(keys[5]) +',' +str(keys[6]) +"\n"
        sample_buffer += sample
        
        if i % 1024:
            write_buffer(sample_buffer, output_dir)
            sample_buffer = ''

        
        i+=1
        loop_time = time.time()-last_time
        #print('loop took {} seconds'.format(loop_time))

        if rate > loop_time:
            sleep_time = rate - loop_time
            #print('sleeping for {} seconds because we have extra time'.format(sleep_time))
            time.sleep(sleep_time)

        last_time = time.time()
    
    write_buffer(sample_buffer, output_dir)





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