import json
import nltk
import collections
import random
import string
from bleu.bleu_scorer import BleuScorer

# get post_data_entry.json, filter punctuation and digist
def get_post_data_entry(raw_data_entry_path, post_data_entry_path):
    with open(raw_data_entry_path, 'r') as file:
        raw_data_dict = json.load(file)

    post_data_dict = {}
    for key in raw_data_dict.keys():
        dict = {}
        raw_data_item = raw_data_dict[key]
        impression_list = raw_data_item['impression']
        findings_list = raw_data_item['findings']

        new_impression_list = []
        filter = string.punctuation + '0123456789'
        for sent_impression in impression_list:
            sentence_re = ''.join(c for c in sent_impression if c not in filter)  # filter out punctuation
            new_impression_list.append(sentence_re)
        new_findings_list = []
        for sent_findings in findings_list:
            sentence_re = ''.join(c for c in sent_findings if c not in filter)  # filter out punctuation
            new_findings_list.append(sentence_re)

        dict['impression'] = new_impression_list
        dict['findings'] = new_findings_list
        post_data_dict[key] = dict

    with open(post_data_entry_path, 'w') as file:
        json.dump(post_data_dict, file)

    print('post data entry get success!')

# get data_entry.json, 3312, filter sentence num < 4
def get_data_entry(post_data_entry_path, data_entry_path):
    with open(post_data_entry_path, 'r') as file:
        post_data_dict = json.load(file)
    print('post data num = %s' % len(post_data_dict))

    data_dict = {}
    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        impression_list = post_data_item['impression']
        findings_list = post_data_item['findings']
        sentence_list = []
        if impression_list.__len__() > 0 and findings_list.__len__() >= 3:
            for sent in impression_list:
                sentence_list.append(sent)
            for sent in findings_list:
                sentence_list.append(sent)
            data_dict[key] = sentence_list

    print('data num = %s' % len(data_dict))

    with open(data_entry_path, 'w') as file:
        json.dump(data_dict, file)
    print('data entry get success!')

def get_vocaluary(data_entry_path, vocabulary_path):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    vocabulary = ['<S>', '</S>', '<EOS>']
    for key in data_dict.keys():
        sent_list = data_dict[key]
        for sent in sent_list:
            word_list = nltk.word_tokenize(sent)
            for word in word_list:
                if not vocabulary.__contains__(word):
                    vocabulary.append(word)
    print('vocabulary size = %s' % len(vocabulary))

    with open(vocabulary_path, 'w') as file:
        json.dump(vocabulary, file)

    print('vocabulary get success!')

def get_split(data_entry_path, train_split_path, test_split_path):
    with open(data_entry_path, 'r') as file:
        data_dict = json.load(file)

    print('all data num = %s' % len(data_dict))

    train_split_list = random.sample(data_dict.keys(), 3000)
    test_split_list = []
    for key in data_dict.keys():
        if not train_split_list.__contains__(key):
            test_split_list.append(key)

    with open(train_split_path, 'w') as file:
        json.dump(train_split_list, file)
    print('test split get success')
    with open(test_split_path, 'w') as file:
        json.dump(test_split_list, file)
    print('train split get success!')


def main():
    raw_data_entry_path = '../../data/data_entry/raw_data_entry.json'
    top_n = 1000
    post_data_entry_path = '../../data/data_entry/post_data_entry.json'
    # get_post_data_entry(raw_data_entry_path, post_data_entry_path)

    # statistics(post_data_entry_path, top_n)
    data_entry_path = '../../data/data_entry/data_entry.json'
    # get_data_entry(post_data_entry_path, data_entry_path)

    vocabulary_path = '../../data/data_entry/vocabulary.json'
    # get_vocaluary(data_entry_path, vocabulary_path)

    train_split_path = '../../data/train_split.json'
    test_split_path = '../../data/test_split.json'
    # get_split(data_entry_path, train_split_path, test_split_path)

main()
