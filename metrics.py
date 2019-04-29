from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import json

def coco_caption_metrics_hier(predicts_list, sentences_list, image_id_list, config, batch_size=26, is_training=True):
    with open(config.vocabulary_path, 'r') as file:
        vocabulary_list = json.load(file)
    word2id = {}
    for i in range(vocabulary_list.__len__()):
        word2id[vocabulary_list[i]] = i
    id2word = {v: k for k, v in word2id.items()}

    gts = {}
    res = {}
    for i in range(0, predicts_list.__len__()):
        for j in range(0, batch_size):
            sent_pre, sent_gt = [], []
            for k in range(config.max_sentence_num * config.max_sentence_length):
                id_input = int(predicts_list[i][k][j])
                sent_pre.append(id2word[id_input])

                id_gt = sentences_list[i][j][k]
                if (not id2word[id_gt].__eq__('</S>')) and (not id2word[id_gt].__eq__('<EOS>')):
                    sent_gt.append(id2word[id_gt])

            # sent_pre2 = sent_pre
            sent_pre2 = []
            for n in range(config.max_sentence_num):
                for m in range(config.max_sentence_length):
                    word = sent_pre[n*config.max_sentence_length + m]
                    if word != '</S>':
                        sent_pre2.append(word)
                    else:
                        break

            str_pre, str_gt = ' '.join(sent_pre2), ' '.join(sent_gt)
            image_id = image_id_list[i][j][0]
            gts[image_id] = [str_gt]
            res[image_id] = [str_pre]

    if not is_training:
        with open(config.result_gts_path, 'w') as file:
            json.dump(gts, file)
        with open(config.result_res_path, 'w') as file:
            json.dump(res, file)

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