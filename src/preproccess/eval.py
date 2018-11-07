from bleu.bleu_scorer import BleuScorer
from bleu.bleu import Bleu
import json
import nltk
import numpy as np

def test():
    predict = 'no acute cardiopulmonary abnormality . the heart size is normal size . there is no pleural effusion or pneumothorax . the lungs are clear . the lungs are clear .'
    with open('../../data/data_entry/post_data_entry.json', 'r') as file:
        post_data_dict = json.load(file)

    gts = {}
    res = {}
    bleu = [0, 0, 0 ,0]
    num = 0
    for key in post_data_dict.keys():
        post_data_item = post_data_dict[key]
        impression_list = post_data_item['impression']
        findings_list = post_data_item['findings']

        word_list = []
        for sent in impression_list:
            word_list += nltk.word_tokenize(sent) + ['.']
        for sent in findings_list[:4]:
            word_list += nltk.word_tokenize(sent) + ['.']
        gt = ' '.join(word_list)

        bleu_scorer = BleuScorer(n=4)
        bleu_scorer += (predict, [gt])
        _bleu, _ = bleu_scorer.compute_score()
        print(key)
        print(gt)
        print(_bleu)
        print('*'*200)
        bleu = np.sum([bleu, _bleu], axis=0)

        num += 1
        if num == 250:
            break

        gts[key] = [gt]
        res[key] = [predict]

    print(bleu / 250)

    bleu_scorer = Bleu(n=4)
    bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)

    print(bleu)

test()