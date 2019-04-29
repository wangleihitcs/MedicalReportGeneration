import json, os
import numpy as np
import random

def get_split(data_entry_path, imgs_dir_path):
    with open(data_entry_path) as f:
        data_dict = json.load(f)
    filenames = os.listdir(imgs_dir_path)

    id_list = []
    for filename in filenames:
        key = filename.split('_')[0]
        if key in data_dict.keys():
            if not key in id_list:
                id_list.append(key)

    test_list = random.sample(id_list, 300)
    train_list = [id for id in id_list if not id in test_list]
    print('all data size = %s' % len(id_list))
    print('train data size = %s' % len(train_list))
    print('test data size = %s' % len(test_list))

    with open('../data/train_split.json', 'w') as f:
        json.dump(train_list, f)
    with open('../data/test_split.json', 'w') as f:
        json.dump(test_list, f)


data_entry_path = '../data/data_entry.json'
imgs_dir_path = '/home/wanglei/workshop/IUX-Ray/NLMCXR_png_pairs'

get_split(data_entry_path, imgs_dir_path)