import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

import datasets
import metrics
from config import Config
# from cnn_hier_rnn_model import Model
from cnn_vis_sem_rnn_model import Model

def train():
    c = Config()
    md = Model(is_training=True, config=c, batch_size=c.batch_size)
    mt = Model(is_training=False, config=c, batch_size=3)

    print('Read Data...')
    image_frontal_batch, image_lateral_batch, sentence_batch, mask_batch, image_id_batch = datasets.get_train_batch(c.train_tfrecord_path, c, md.batch_size)
    image_frontal_batch2, image_lateral_batch2, sentence_batch2, mask_batch2, image_id_batch2 = datasets.get_train_batch(c.test_tfrecord_path, c, mt.batch_size)

    init_fn_frontal = slim.assign_from_checkpoint_fn(c.pretrain_cnn_model_frontal, slim.get_model_variables('FrontalInceptionV3'))
    init_fn_lateral = slim.assign_from_checkpoint_fn(c.pretrain_cnn_model_lateral, slim.get_model_variables('LateralInceptionV3'))

    saver = tf.train.Saver(max_to_keep=100)
    print('Train Model...')
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(c.summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        init_fn_frontal(sess)
        init_fn_lateral(sess)

        coord = tf.train.Coordinator()  # queue manage
        threads = tf.train.start_queue_runners(coord=coord)

        iter = 0
        # loss_list, acc_list, predicts_list, sentences_list, image_id_list = [], [], [], [], []
        for epoch in range(c.epoch_num):
            for _ in range(c.train_num / md.batch_size):
                images_frontal, images_lateral, sentences, masks, image_ids = sess.run([image_frontal_batch, image_lateral_batch, sentence_batch, mask_batch, image_id_batch])
                feed_dict = {
                    md.images_frontal: images_frontal,
                    md.images_lateral: images_lateral,
                    md.sentences: sentences,
                    md.masks: masks
                }
                _, _summary, _global_step, _loss, _acc, _predicts, = sess.run(
                    [md.step_op, md.summary, md.global_step, md.loss, md.accuracy, md.predicts], feed_dict=feed_dict)
                train_writer.add_summary(_summary, _global_step)

                loss_list.append(_loss)
                acc_list.append(_acc)
                predicts_list.append(_predicts)
                sentences_list.append(sentences)
                image_id_list.append(image_ids)

                iter += 1
                if iter % 100 == 0:
                    # train test
                    # bleu, meteor, rouge, cider = metrics.coco_caption_metrics_hier(predicts_list,
                    #                                                                sentences_list,
                    #                                                                image_id_list,
                    #                                                                config=c,
                    #                                                                batch_size=md.batch_size,
                    #                                                                is_training=md.is_training)
                    # print('iter = %s, loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                    #       (iter, np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))

                    # test test
                    loss_list, acc_list, predicts_list, sentences_list, image_id_list = [], [], [], [], []
                    for _ in range(c.test_num / mt.batch_size):
                        images_frontal, images_lateral, sentences, masks, image_ids = sess.run([image_frontal_batch2, image_lateral_batch2, sentence_batch2, mask_batch2, image_id_batch2])
                        feed_dict = {
                            mt.images_frontal: images_frontal,
                            mt.images_lateral: images_lateral,
                            mt.sentences: sentences,
                            mt.masks: masks
                        }
                        _loss, _acc, _predicts = sess.run([mt.loss, mt.accuracy, mt.predicts], feed_dict=feed_dict)
                        loss_list.append(_loss)
                        acc_list.append(_acc)
                        predicts_list.append(_predicts)
                        sentences_list.append(sentences)
                        image_id_list.append(image_ids)

                    bleu, meteor, rouge, cider = metrics.coco_caption_metrics_hier(predicts_list,
                                                                                   sentences_list,
                                                                                   image_id_list,
                                                                                   config=c,
                                                                                   batch_size=mt.batch_size,
                                                                                   is_training=mt.is_training)
                    print('---------iter = %s, loss = %.4f, acc = %.4f, bleu = %s, meteor = %s, rouge = %s, cider = %s' %
                          (iter, np.mean(loss_list), np.mean(acc_list), bleu, meteor, rouge, cider))
                    loss_list, acc_list, predicts_list, sentences_list, image_id_list = [], [], [], [], []

                    saver.save(sess, c.model_path, global_step=iter)

        coord.request_stop()
        coord.join(threads)


train()