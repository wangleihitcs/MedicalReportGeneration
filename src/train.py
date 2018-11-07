import tensorflow as tf
from tensorflow.contrib import slim
import termcolor as tc
import numpy as np

import datasets
import metrics
# from cnn_rnn_model import Model
# from cnn_rnn_att_model import Model
# from cnn_hier_rnn_model import Model
# from cnn_hier_rnn_vgg_model import Model
# from cnn_hier_rnn_resnet_model import Model
from cnn_1d_conv_model import Model
# from cnn_1d_conv_att_model import Model

imgs_dir_path = '/home/wanglei/Documents/IUX-Ray/front_view_data'
data_entry_path = '../data/data_entry/data_entry.json'
train_split_path = '../data/train_split.json'
val_split_path = '../data/test_split.json'
vocabulary_path = '../data/data_entry/vocabulary.json'
pretrain_inception_v3_ckpt_path = '/home/wanglei/Documents/b_pre_train_model/inception/inception_v3.ckpt'
pretrain_vgg_19_ckpt_path = '/home/wanglei/Documents/b_pre_train_model/vgg/vgg_19.ckpt'
pretrain_resnet_152_ckpt_path = '/home/wanglei/Documents/b_pre_train_model/resnet/resnet_v2_152.ckpt'

start_epoch = 0
end_epoch = 31
summary_path = '../data/summary'
model_path = '../data/model/my-test-300'
model_path_save = '../data/model/my-test'
def train():
    md = Model(is_training=True) # Train model
    mdv = Model(is_training=True)  # Train model

    print('---Read Dataset...')
    image_list_t, sentence_list_t, mask_list_t, filename_list_t = datasets.get_input_hier(imgs_dir_path, data_entry_path,
                                                                                    train_split_path, vocabulary_path)
    train_num = image_list_t.__len__()
    print('train num = %s' % train_num)
    image_list_v, sentence_list_v, mask_list_v, filename_list_v = datasets.get_input_hier(imgs_dir_path, data_entry_path,
                                                                                     val_split_path, vocabulary_path)
    val_num = image_list_v.__len__()
    print('val num = %s' % val_num)

    print('---Training Model...')
    init_fn = slim.assign_from_checkpoint_fn(pretrain_inception_v3_ckpt_path, slim.get_model_variables('InceptionV3'))  # 'vgg_19' or 'InceptionV3'
    # init_fn = slim.assign_from_checkpoint_fn(pretrain_vgg_19_ckpt_path, slim.get_model_variables('vgg_19'))  # 'vgg_19' or 'InceptionV3'
    # init_fn = slim.assign_from_checkpoint_fn(pretrain_resnet_152_ckpt_path, slim.get_model_variables('resnet_v2_152'))  # 'vgg_19' or 'InceptionV3'

    saver = tf.train.Saver(max_to_keep=301)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        init_fn(sess)
        # saver.restore(sess, model_path)

        for i in range(start_epoch, end_epoch):
            loss_list, acc_list, predictions_list = [], [], []
            sentence_list_metrics = []

            iters = int(train_num / md.batch_size)
            for k in range(iters):
                images = image_list_t[k * md.batch_size:(k + 1) * md.batch_size]
                sentences = sentence_list_t[k * md.batch_size:(k + 1) * md.batch_size]
                masks = mask_list_t[k * md.batch_size:(k + 1) * md.batch_size]
                feed_dict = {md.images: images, md.sentences: sentences,
                             md.masks: masks}
                _, _summary, _global_step, _loss, _acc, _predictions, = sess.run(
                    [md.step_op, md.summary, md.global_step, md.loss, md.accuracy, md.predictions], feed_dict=feed_dict)
                train_writer.add_summary(_summary, _global_step)

                loss_list.append(_loss)
                acc_list.append(_acc)
                predictions_list.append(_predictions)
                sentence_list_metrics += sentences

            if i % 1 == 0:
                bleu, meteor, rouge, cider = metrics.coco_caption_metrics_hier(predictions_list, sentence_list_metrics, filename_list_t, vocabulary_path, batch_size=md.batch_size)
                print('epoch = %s, loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                      (i, np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

                loss_list, acc_list, predictions_list = [], [], []
                sentence_list_metrics = []
                iters = int(val_num / md.batch_size)
                for k in range(iters):
                    images = image_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    sentences = sentence_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    masks = mask_list_v[k * md.batch_size:(k + 1) * md.batch_size]
                    feed_dict = {mdv.images: images, mdv.sentences: sentences,
                                 mdv.masks: masks}
                    _loss, _acc, _predictions, = sess.run(
                        [mdv.loss, mdv.accuracy, mdv.predictions],
                        feed_dict=feed_dict)

                    loss_list.append(_loss)
                    acc_list.append(_acc)
                    predictions_list.append(_predictions)
                    sentence_list_metrics += sentences

                bleu, meteor, rouge, cider = metrics.coco_caption_metrics_hier(predictions_list, sentence_list_metrics, filename_list_v, vocabulary_path, batch_size=mdv.batch_size)
                loss_val = round(np.mean(loss_list), 4)
                print('------epoch = %s, loss = %s, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                      (i, tc.colored(loss_val, 'red'), np.mean(acc_list), bleu, meteor, rouge, cider))

            if i % 1 == 0:
                saver.save(sess, model_path_save, global_step=i)

        train_writer.close()
        print('---Training complete.')
train()
