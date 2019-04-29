import tensorflow as tf
import json, nltk, os
import numpy as np

import config
from utils import image_utils

def get_train_batch(tfrecord_path, config, batch_size=26):
    tfrecord_path_list = [tfrecord_path]

    # 1. get filename_queue
    filename_queue = tf.train.string_input_producer(tfrecord_path_list, shuffle=False)

    # 2. get image pixels, sentence, mask, image_id
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_frontal_pixels': tf.FixedLenFeature([config.image_size * config.image_size * 3], tf.float32),
            'image_lateral_pixels': tf.FixedLenFeature([config.image_size * config.image_size * 3], tf.float32),
            'sentence': tf.FixedLenFeature([config.max_sentence_num*config.max_sentence_length], tf.int64),
            'mask': tf.FixedLenFeature([config.max_sentence_num*config.max_sentence_length], tf.int64),
            'image_id': tf.FixedLenFeature([1], tf.int64),
        }
    )
    image_frontal = tf.reshape(features['image_frontal_pixels'], [config.image_size, config.image_size, 3])
    image_lateral = tf.reshape(features['image_lateral_pixels'], [config.image_size, config.image_size, 3])
    sentence = features['sentence']
    mask = features['mask']
    image_id = features['image_id']

    # 3. get tf.tfrecord.batch
    image_frontal_batch, image_lateral_batch, sentece_batch, mask_batch, image_id_batch = tf.train.shuffle_batch(
        [image_frontal, image_lateral, sentence, mask, image_id],
        batch_size=batch_size,
        capacity=3 * batch_size,
        min_after_dequeue=2 * batch_size
    )

    return image_frontal_batch, image_lateral_batch, sentece_batch, mask_batch, image_id_batch

def get_train_tfrecord(imgs_path, data_entry_path, split_list_path, vocabulary_path, tfrecord_path, config):
    with open(vocabulary_path, 'r') as f:
        vocabulary = json.load(f)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i

    filenames = os.listdir(imgs_path)
    with open(data_entry_path, 'r') as f:
        data_dict = json.load(f)
    with open(split_list_path, 'r') as f:
        split_id_list = json.load(f)

    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    for id in split_id_list:
        two_name = []
        for filename in filenames:
            if id == filename.split('_')[0]:
                two_name.append(filename)

        frontal_image_name, lateral_image_name = two_name[0], two_name[1]
        if two_name[0] > two_name[1]:
            frontal_image_name, lateral_image_name = two_name[1], two_name[0]

        image_frontal = image_utils.getImages(os.path.join(imgs_path, frontal_image_name), config.image_size)
        image_frontal = image_frontal.reshape([config.image_size*config.image_size*3])
        image_lateral = image_utils.getImages(os.path.join(imgs_path, lateral_image_name), config.image_size)
        image_lateral = image_lateral.reshape([config.image_size*config.image_size*3])

        sent_list = data_dict[id]
        if sent_list.__len__() > config.max_sentence_num:
            sent_list = sent_list[:config.max_sentence_num]

        word_list = []
        for sent in sent_list:
            words = nltk.word_tokenize(sent)
            if words.__len__() >= config.max_sentence_length:
                for i in range(config.max_sentence_length - 1):
                    word_list.append(words[i])
                word_list.append('</S>')
            else:
                for i in range(words.__len__()):
                    word_list.append(words[i])
                word_list.append('</S>')
                for _ in range(config.max_sentence_length - words.__len__() - 1):
                    word_list.append('<EOS>')
        for _ in range(config.max_sentence_num - sent_list.__len__()):
            word_list.append('</S>')
            for _ in range(config.max_sentence_length-1):
                word_list.append('<EOS>')
        # print(word_list.__len__())

        sentence = np.zeros(shape=[config.max_sentence_num * config.max_sentence_length], dtype=np.int64)
        mask = np.ones(shape=[config.max_sentence_num * config.max_sentence_length], dtype=np.int64)
        for i in range(config.max_sentence_num*config.max_sentence_length):
            sentence[i] = word2id[word_list[i]]
            if word_list == '<EOS>':
                mask[i] = 0

        image_id = int(id[3:])
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_frontal_pixels': tf.train.Feature(float_list=tf.train.FloatList(value=image_frontal)),
                    'image_lateral_pixels': tf.train.Feature(float_list=tf.train.FloatList(value=image_lateral)),
                    'sentence': tf.train.Feature(int64_list=tf.train.Int64List(value=sentence)),
                    'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=mask)),
                    'image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_id]))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)

    print('%s write to tfrecord success!' % tfrecord_path)

# config = config.Config()
# #1. get train.tfrecord
# get_train_tfrecord(config.imgs_dir_path, config.data_entry_path, config.train_list_path, config.vocabulary_path, config.train_tfrecord_path, config)
#
# #2. get test.tfrecord
# get_train_tfrecord(config.imgs_dir_path, config.data_entry_path, config.test_list_path, config.vocabulary_path, config.test_tfrecord_path, config)

