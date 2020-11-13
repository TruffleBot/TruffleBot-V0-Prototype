#%%
import re
import tensorflow as tf

FILE_PATTERN = 'D:/wowrecords/' + '*.tfrec'

def int_sort(name):
    return int(re.sub(r'\D', '', name))

def get_filelists():

    names = tf.io.gfile.glob(FILE_PATTERN)
    names.sort(key=int_sort)
    
    evan1_files = [name for name in names if 'evan1' in name]
    evan2_files = [name for name in names if 'evan2' in name]
    loom_files = [name for name in names if 'loom' in name]
    phil_files = [name for name in names if 'phil' in name]
    steve_files = [name for name in names if 'steve' in name]

    phil_1_files = phil_files[:82]
    phil_2_files = phil_files[82:]

    steve_1_files = steve_files[59:]
    steve_2_files = steve_files[:59]

    rest_names = set(names) - set(evan1_files) - set(evan2_files) - set(loom_files) - set(phil_files) - set(steve_files)
    rest_names = list(rest_names)
    rest_names.sort(key=int_sort)

    og_files = rest_names

    files_list = [evan1_files, evan2_files, loom_files, phil_1_files, phil_2_files, steve_1_files, steve_2_files, og_files]
    val_files = list()

    total = 0

    for i,l in enumerate(files_list):
        files_list[i] = files_list[i][:-3]
        val_files.append(files_list[i][-3:])
        total += len(files_list[i])
        print( l[0].strip().split('/')[-1].split('.')[0]+ ': '+str(len(files_list[i])))
    print(total)
    return files_list, val_files
