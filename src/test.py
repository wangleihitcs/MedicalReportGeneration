import tensorflow as tf
import numpy as np

import datasets
import metrics

# from cnn_rnn_model import Model
# from cnn_rnn_att_model import Model
# from cnn_hier_rnn_model import Model
# from cnn_1d_conv_model import Model
from cnn_1d_conv_only_semantic_model import Model
# from cnn_1d_conv_only_visual_model import Model

imgs_dir_path = '/home/wanglei/Documents/IUX-Ray/front_view_data'
data_entry_path = '../data/data_entry/data_entry.json'
test_split_path = '../data/test_split.json'
vocabulary_path = '../data/data_entry/vocabulary.json'

model_path = '../data/model/my-test-100'
def test():
    md = Model(is_training=False)

    print('---Read Dataset...')
    image_list, sentence_list, mask_list, filename_list = datasets.get_input_hier(imgs_dir_path, data_entry_path,
                                                                            test_split_path, vocabulary_path)
    test_num = image_list.__len__()
    print('test num = %s' % test_num)

    print('---Testing Model...')
    saver = tf.train.Saver(max_to_keep=300)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

        loss_list, acc_list, predictions_list = [], [], []
        sentence_list_metrics = []
        iters = int(test_num / md.batch_size)
        for k in range(iters):
            images = image_list[k * md.batch_size:(k + 1) * md.batch_size]
            sentences = sentence_list[k * md.batch_size:(k + 1) * md.batch_size]
            masks = mask_list[k * md.batch_size:(k + 1) * md.batch_size]
            feed_dict = {md.images: images, md.sentences: sentences,
                         md.masks: masks}
            _loss, _acc, _predictions, = sess.run(
                [md.loss, md.accuracy, md.predictions],
                feed_dict=feed_dict)

            loss_list.append(_loss)
            acc_list.append(_acc)
            predictions_list.append(_predictions)
            sentence_list_metrics += sentences
        bleu, meteor, rouge, cider = metrics.coco_caption_metrics_hier1(predictions_list, sentence_list_metrics,
                                                                    filename_list, vocabulary_path, batch_size=md.batch_size, is_training=False)
        print('loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
              (np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

        print(md.is_train)
        print(model_path)
        print(test_num)
    print("---Test complete.")

test()
