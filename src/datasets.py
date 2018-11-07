import numpy as np
import os
import json
import nltk
# nltk.download('punkt')
from PIL import Image

# get vocabulary from ../data/data_entry/vocabulary.json, size = 1966
def get_vocabulary(vocabulary_path):
    with open(vocabulary_path, 'r') as file:
        vocabulary = json.load(file)
    return vocabulary

def get_input(imgs_dir_path, data_entry_path, split_json_path, vocabulary_path, image_size=224, max_caption_length=100):
    vocabulary = get_vocabulary(vocabulary_path)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i

    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    with open(split_json_path, 'r') as file:
        id_list = json.load(file)

    image_list, sentence_list, mask_list, filename_list = [], [], [], []
    filenames = os.listdir(imgs_dir_path)
    for filename in filenames:
        id = filename.split('_')[0]
        if id in id_list and id in data_dict.keys():
            img_path = os.path.join(imgs_dir_path, filename)
            image_list.append(getImages(img_path, image_size))

            sent_list = data_dict[id]
            sentence = []
            for sent in sent_list:
                word_list = nltk.word_tokenize(sent)
                sentence += word_list
            if len(sentence) < max_caption_length:
                sentence.append('</S>')
                for _ in range(max_caption_length - len(sentence)):
                    sentence.append('<EOS>')
            else:
                sentence = sentence[:max_caption_length]
            sentence_list.append(getSentence(word2id, sentence, max_caption_length))

            mask_list.append(getMask(sentence, max_caption_length))

            filename_list.append(id)

    return image_list, sentence_list, mask_list, filename_list

def get_input_hier(imgs_dir_path, data_entry_path, split_json_path, vocabulary_path, image_size=224, max_caption_length=300, N=6, M=50):
    vocabulary = get_vocabulary(vocabulary_path)
    word2id = {}
    for i in range(vocabulary.__len__()):
        word2id[vocabulary[i]] = i

    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    with open(split_json_path, 'r') as file:
        id_list = json.load(file)

    image_list, sentence_list, mask_list, filename_list = [], [], [], []
    filenames = os.listdir(imgs_dir_path)
    for filename in filenames:
        id = filename.split('_')[0]
        if id in id_list and id in data_dict.keys():
            img_path = os.path.join(imgs_dir_path, filename)
            image_list.append(getImages(img_path, image_size))

            sent_list = data_dict[id]
            sentence = []
            if sent_list.__len__() < N:
                for sent in sent_list:
                    word_list = nltk.word_tokenize(sent)
                    if word_list.__len__() < M:
                        word_list.append('</S>')
                        for _ in range(M - word_list.__len__()):
                            word_list.append('<EOS>')
                    else:
                        word_list = word_list[:M]
                    sentence += word_list
                for _ in range(N - sent_list.__len__()):
                    word_list = []
                    word_list.append('</S>')
                    for _ in range(M - word_list.__len__()):
                        word_list.append('<EOS>')
                    sentence += word_list
            else:
                for sent in sent_list[:N]:
                    word_list = nltk.word_tokenize(sent)
                    if word_list.__len__() < M:
                        word_list.append('</S>')
                        for _ in range(M - word_list.__len__()):
                            word_list.append('<EOS>')
                    else:
                        word_list = word_list[:M]
                    sentence += word_list
            sentence_list.append(getSentence(word2id, sentence, max_caption_length))

            mask_list.append(getMask(sentence, max_caption_length))

            filename_list.append(id)

    return image_list, sentence_list, mask_list, filename_list

# get image pixels value
def getImages(image_path, image_size):
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image = image_rgb.resize((image_size, image_size), Image.ANTIALIAS)
    image_array = np.array(image) / 255.0
    return image_array

# get sentence id in vocabulary
def getSentence(word2id, sentence, max_caption_length):
    sentence_id = np.zeros([max_caption_length])
    for i in range(max_caption_length):
        sentence_id[i] = word2id[sentence[i]]
    return sentence_id

def getMask(sentence, max_caption_length):
    mask = np.zeros([max_caption_length])
    for i in range(max_caption_length):
        if sentence[i] != '<EOS>':
            mask[i] = 1
    return mask