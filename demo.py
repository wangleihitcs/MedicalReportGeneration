import tensorflow as tf
import numpy as np
import json

from config import Config
from utils import image_utils
from cnn_hier_rnn_model import Model
# from cnn_sem_rnn_model import Model

def get_test_data(img_frontal_path, img_lateral_path, config):
    image_frontal = np.zeros([1, config.image_size, config.image_size, 3])
    image_frontal[0] = image_utils.getImages(img_frontal_path, config.image_size)
    image_lateral = np.zeros([1, config.image_size, config.image_size, 3])
    image_lateral[0] = image_utils.getImages(img_lateral_path, config.image_size)

    sentence = np.zeros([1, config.max_sentence_num * config.max_sentence_length])
    mask = np.zeros([1, config.max_sentence_num * config.max_sentence_length])

    return image_frontal, image_lateral, sentence, mask

def get_sentences(predicts_list, config):
    with open(config.vocabulary_path, 'r') as f:
        vocabulary_list = json.load(f)
    word2id = {}
    for i in range(vocabulary_list.__len__()):
        word2id[vocabulary_list[i]] = i
    id2word = {v: k for k, v in word2id.items()}

    sentence_list = []
    for i in range(config.max_sentence_num):
        sentence = []
        for j in range(config.max_sentence_length):
            id = int(predicts_list[0][i*config.max_sentence_length + j][0])
            if id2word[id] == '</S>':
                break
            else:
                sentence.append(id2word[id])
        sentence = ' '.join(sentence)
        sentence_list.append(sentence)
    return sentence_list

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string('img_frontal_path', './data/experiments/CXR1900_IM-0584-1001.png', 'The frontal image path')
tf.flags.DEFINE_string('img_lateral_path', './data/experiments/CXR1900_IM-0584-2001.png', 'The lateral image path')
tf.flags.DEFINE_string('model_path', './data/model/my-test-1000', 'The test model path')


img_frontal_path = FLAGS.img_frontal_path
img_lateral_path = FLAGS.img_lateral_path
model_path = FLAGS.model_path

config = Config()
mt = Model(is_training=False, batch_size=1)

img_frontal, img_lateral, sentence, mask = get_test_data(img_frontal_path, img_lateral_path, config)

saver = tf.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    feed_dict = {
        mt.images_frontal: img_frontal,
        mt.images_lateral: img_lateral,
        mt.sentences: sentence,
        mt.masks: mask
    }
    predicts_list = sess.run([mt.predicts], feed_dict=feed_dict)

sentence_list = get_sentences(predicts_list, config)

print('The generate report:')
for sentence in sentence_list:
    print('\t %s' % sentence)