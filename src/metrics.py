from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import json

# bleu, meteor, rouge, cider
def coco_caption_metrics(predictions_list, sentences_list, filename_list, vocabulary_path, max_caption_length=100, batch_size=30, is_training=True):
    with open(vocabulary_path, 'r') as file:
        vocabulary_list = json.load(file)
    word2id = {}
    for i in range(vocabulary_list.__len__()):
        word2id[vocabulary_list[i]] = i
    id2word = {v: k for k, v in word2id.items()}

    gts = {}
    res = {}
    for i in range(0, predictions_list.__len__()):
        for j in range(0, batch_size):
            sen_input, sen_ground_truth = [], []
            for k in range(max_caption_length):
                id_input = int(predictions_list[i][k][j])
                sen_input.append(id2word[id_input])

                id_ground_truth = sentences_list[(i*batch_size)+j][k]
                if (not id2word[id_ground_truth].__eq__('</S>')) and (not id2word[id_ground_truth].__eq__('<EOS>')):
                    sen_ground_truth.append(id2word[id_ground_truth])

            sen_pre = []
            for n in range(max_caption_length):
                word = sen_input[n]
                if word != '</S>':
                    sen_pre.append(word)
                else:
                    break

            str_input, str_grundtruth = ' '.join(sen_pre), ' '.join(sen_ground_truth)
            filename = filename_list[(i * batch_size) + j]
            gts[filename] = [str_grundtruth]
            res[filename] = [str_input]

    if not is_training:
        for key in gts.keys():
            str_input = res[key][0]
            str_grundtruth = gts[key][0]
            print(key)
            print(str_input)
            print(str_grundtruth)
            print('*' * 100)

    bleu_scorer = Bleu(n=4)
    bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)

    rouge_scorer = Rouge()
    rouge, _ = rouge_scorer.compute_score(gts=gts, res=res)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts=gts, res=res)

    meteor_scorer = Meteor()
    meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)

    for i in range(4):
        bleu[i] = round(bleu[i], 4)
    return bleu, round(meteor, 4), round(rouge, 4), round(cider, 4)

def coco_caption_metrics_hier(predictions_list, sentences_list, filename_list, vocabulary_path, max_caption_length=300, batch_size=64, N=6, M=50, is_training=True):
    with open(vocabulary_path, 'r') as file:
        vocabulary_list = json.load(file)
    word2id = {}
    for i in range(vocabulary_list.__len__()):
        word2id[vocabulary_list[i]] = i
    id2word = {v: k for k, v in word2id.items()}

    gts = {}
    res = {}

    for i in range(0, predictions_list.__len__()):
        for j in range(0, batch_size):
            sen_input, sen_ground_truth = [], []
            for k in range(max_caption_length):
                id_input = int(predictions_list[i][k][j])
                sen_input.append(id2word[id_input])

                id_ground_truth = sentences_list[(i*batch_size)+j][k]
                if (not id2word[id_ground_truth].__eq__('</S>')) and (not id2word[id_ground_truth].__eq__('<EOS>')):
                    sen_ground_truth.append(id2word[id_ground_truth])
                # if (not id2word[id_ground_truth].__eq__('<EOS>')):
                #     sen_ground_truth.append(id2word[id_ground_truth])

            sen_pre = []
            for n in range(N):
                for m in range(M):
                    word = sen_input[n*M + m]
                    if word != '</S>':
                        sen_pre.append(word)
                    else:
                        # sen_pre.append('</S>')
                        break

            str_input, str_grundtruth = ' '.join(sen_pre), ' '.join(sen_ground_truth)
            filename = filename_list[(i * batch_size) + j]
            gts[filename] = [str_grundtruth]
            res[filename] = [str_input]


    # print(gts)
    # print(res)
    if not is_training:
        for key in gts.keys():
            str_input = res[key][0]
            str_grundtruth = gts[key][0]
            print(key)
            print(str_input)
            print(str_grundtruth)
            print('*' * 100)

        with open('../data/result_res.json', 'w') as file:
            json.dump(res, file)
        with open('../data/result_gts.json', 'w') as file:
            json.dump(gts, file)
        print('result.json get success')

    bleu_scorer = Bleu(n=4)
    bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)

    rouge_scorer = Rouge()
    rouge, _ = rouge_scorer.compute_score(gts=gts, res=res)

    cider_scorer = Cider()
    cider, _ = cider_scorer.compute_score(gts=gts, res=res)

    meteor_scorer = Meteor()
    meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)

    for i in range(4):
        bleu[i] = round(bleu[i], 4)
    return bleu, round(meteor, 4), round(rouge, 4), round(cider, 4)

def coco_eval(result_gts_path, result_res_path):

    with open(result_gts_path, 'r') as file:
        gt_dict = json.load(file)
    with open(result_res_path, 'r') as file:
        res_dict = json.load(file)

    _bleu, _meteor, _rouge, _cider = [0,0,0,0], 0, 0, 0
    for key in gt_dict.keys():
        gts = {key:gt_dict[key]}
        res = {key:res_dict[key]}

        bleu_scorer = Bleu(n=4)
        bleu, _ = bleu_scorer.compute_score(gts=gts, res=res)
        for i in range(4):
            _bleu[i] += bleu[i]

        # rouge_scorer = Rouge()
        # rouge, _ = rouge_scorer.compute_score(gts=gts, res=res)
        # _rouge += rouge
        #
        # cider_scorer = Cider()
        # cider, _ = cider_scorer.compute_score(gts=gts, res=res)
        # _cider += cider
        #
        # meteor_scorer = Meteor()
        # meteor, _ = meteor_scorer.compute_score(gts=gts, res=res)
        # _meteor += meteor
        meteor, rouge, cider = 0, 0, 0
        print(key)
        print('bleu = %s, meteor = %s, rouge = %s, cider = %s' % (bleu, meteor, rouge, cider))
        print('*'*100)

    for i in range(4):
        _bleu[i] = _bleu[i]/len(gt_dict)
    _meteor = _meteor/len(gt_dict)
    _rouge = _rouge/len(gt_dict)
    _cider = _cider/len(gt_dict)
    print('all result mean...')
    print('bleu = %s, meteor = %s, rouge = %s, cider = %s' % (_bleu, _meteor, _rouge, _cider))

def main():
    result_gts_path = '../data/result_gts.json'
    result_res_path = '../data/result_res.json'
    coco_eval(result_gts_path, result_res_path)
# main()