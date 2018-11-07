import json
import nltk
import collections
import os
import string
from bleu.bleu_scorer import BleuScorer

def template_impression(post_data_entry_path, template_path):
    with open(post_data_entry_path, 'r') as file:
        post_data_dict = json.load(file)

    # 1. all sentence 7139
    sentence_list_all = []
    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        impression_list = post_data_item['impression']
        for sent_impression in impression_list:
            sentence_list_all.append(sent_impression)
    print('sentences num = %s' % len(sentence_list_all))

    # 2. remove repeated sentence 3086
    sentence_list_post = []
    for sent in sentence_list_all:
        word_list = nltk.word_tokenize(sent)
        sent = ' '.join(word_list)
        if not sentence_list_post.__contains__(sent):
            sentence_list_post.append(sent)
            # print(sent)
    print('sentences remove repeated num = %s' % len(sentence_list_post))

    cluster = {}
    for sent_post in sentence_list_post:
        cluster[sent_post] = 0

    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        impression_list = post_data_item['impression']
        for sent_impression in impression_list:
            for sent_post in sentence_list_post:
                bleu_scorer = BleuScorer(n=4)
                bleu_scorer += (sent_impression, [sent_post])
                bleu, _ = bleu_scorer.compute_score()
                if bleu[0] > 0.8:
                    cluster[sent_post] += 1
        print('%s compare success!' % key)

    for sent_post in cluster.keys():
        print(sent_post, cluster[sent_post])

    with open(template_path, 'w') as file:
        json.dump(cluster, file)
    print('template impression get success!')

def template_findings(post_data_entry_path, template_path):
    with open(post_data_entry_path, 'r') as file:
        post_data_dict = json.load(file)

    # 1. all sentence 7139
    sentence_list_all = []
    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        findings_list = post_data_item['findings']
        for sent_impression in findings_list:
            sentence_list_all.append(sent_impression)
    print('sentences num = %s' % len(sentence_list_all))

    # 2. remove repeated sentence 5639
    sentence_list_post = []
    for sent in sentence_list_all:
        word_list = nltk.word_tokenize(sent)
        sent = ' '.join(word_list)
        if not sentence_list_post.__contains__(sent):
            sentence_list_post.append(sent)
            # print(sent)
    print('sentences remove repeated num = %s' % len(sentence_list_post))

    cluster = {}
    for sent_post in sentence_list_post:
        cluster[sent_post] = 0

    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        findings_list = post_data_item['findings']
        for sent_impression in findings_list:
            for sent_post in sentence_list_post:
                bleu_scorer = BleuScorer(n=4)
                bleu_scorer += (sent_impression, [sent_post])
                bleu, _ = bleu_scorer.compute_score()
                if bleu[0] > 0.8:
                    cluster[sent_post] += 1
        print('%s compare success!' % key)

    for sent_post in cluster.keys():
        print(sent_post, cluster[sent_post])

    with open(template_path, 'w') as file:
        json.dump(cluster, file)
    print('template findings get success!')

def template_statistic(template_path):
    with open(template_path, 'r') as file:
        template_dict = json.load(file)

    time = 50
    time_num = 0

    items = sorted(template_dict.items(), lambda x, y:cmp(x[1], y[1]), reverse=True) # sorted by value

    for key, value in items:
        if value > time:
            time_num += 1
            print(key, value)
    print('sentence occur > %s num = %s' % (time, time_num))


def main():
    raw_data_entry_path = '../../data/data_entry/raw_data_entry.json'
    top_n = 1000
    post_data_entry_path = '../../data/data_entry/post_data_entry.json'
    # get_post_data_entry(raw_data_entry_path, post_data_entry_path)
    template_path1 = '../../data/template_impression.json'
    # template_impression(post_data_entry_path, template_path1)
    template_path2 = '../../data/template_findings.json'
    # template_findings(post_data_entry_path, template_path2)

    template_statistic(template_path2)
main()
