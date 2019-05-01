class Config(object):
    def __init__(self):
        self.imgs_dir_path = './data/NLMCXR_png_pairs'
        self.data_entry_path = './data/data_entry.json'
        self.train_list_path = './data/train_split.json'
        self.test_list_path = './data/test_split.json'
        self.vocabulary_path = './data/vocabulary.json'
        self.train_tfrecord_path = './data/tfrecords/train.tfrecord'
        self.test_tfrecord_path = './data/tfrecords/test.tfrecord'
        self.pretrain_cnn_model_frontal = './data/pretrain_model/frontal_inception_v3.ckpt'
        self.pretrain_cnn_model_lateral = './data/pretrain_model/lateral_inception_v3.ckpt'
        self.summary_path = './data/summary/'
        self.model_path = './data/model/my-test'
        self.result_res_path = './data/result/res.json'
        self.result_gts_path = './data/result/gts.json'

        self.batch_size = 26
        self.vocabulary_size = 2068
        self.rnn_units = 512
        self.word_embedding_size = 512
        self.image_size = 299
        self.max_sentence_num = 8
        self.max_sentence_length = 50
        self.epoch_num = 50
        self.train_num = 2761
        self.test_num = 350

        self.learning_rate = 1e-4
        self.dropout_rate = 0.5
        self.rnn_dropout_rate = 0.3
        self.decay_iters = 5 * self.train_num / self.batch_size
        self.decay_rate = 0.9
