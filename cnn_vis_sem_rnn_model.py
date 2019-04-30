import tensorflow as tf
from nets import inception

class Model(object):
    def __init__(self, config, is_training=True, batch_size=26):
        self.config = config
        self.is_training = is_training
        self.batch_size = batch_size
        self.images_frontal = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, config.image_size, config.image_size, 3])
        self.images_lateral = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, config.image_size, config.image_size, 3])
        self.sentences = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, config.max_sentence_num * config.max_sentence_length])
        self.masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, config.max_sentence_num * config.max_sentence_length])

        self.build_cnn()
        self.build_rnn()
        self.build_metrics()
        if is_training:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        net_f, _ = inception.inception_v3(self.images_frontal, trainable=True, is_training=True, add_summaries=False, scope='FrontalInceptionV3')
        net_l, _ = inception.inception_v3(self.images_lateral, trainable=True, is_training=True, add_summaries=False, scope='LateralInceptionV3')

        self.visual_feats = tf.concat([net_f, net_l], axis=1) # [batch_size, 4096]
        print('cnn built.')

    def build_rnn(self):
        with tf.variable_scope("word_embedding"):
            word_embedding_matrix = tf.get_variable(
                                    name='weights',
                                    shape=[self.config.vocabulary_size, self.config.word_embedding_size],
                                    trainable=True)

        # 1. build rnn
        WordRNN = tf.nn.rnn_cell.LSTMCell(
            name='word_rnn',
            num_units=self.config.rnn_units)
        if self.is_training:
            WordRNN = tf.nn.rnn_cell.DropoutWrapper(
                WordRNN,
                input_keep_prob=1.0 - self.config.rnn_dropout_rate,
                output_keep_prob=1.0 - self.config.rnn_dropout_rate,
                state_keep_prob=1.0 - self.config.rnn_dropout_rate)

        predicts = []
        cross_entropies = []
        corrects = []
        global last_sentence
        # 2. generate first sentence
        for sent_id in range(1):
            # 2.1 init Word RNN
            with tf.variable_scope('word_rnn_initialize_0'):
                context = self.visual_feats
                init_c = tf.layers.dense(context, units=self.config.rnn_units, activation=tf.tanh, use_bias=True, name='fc_c')
                init_h = tf.layers.dense(context, units=self.config.rnn_units, activation=tf.tanh, use_bias=True, name='fc_h')

                WordRNN_last_state = init_c, init_h
                WordRNN_last_word = tf.zeros([self.batch_size], tf.int32)

            # 2.2 generate word one by one
            last_sentence = []
            for word_id in range(self.config.max_sentence_length):
                with tf.variable_scope('word_embedding'):
                    word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, WordRNN_last_word)

                with tf.variable_scope('word_rnn'):
                    WordRNN_output, WordRNN_state = WordRNN(word_embedding, WordRNN_last_state)
                    WordRNN_last_state = WordRNN_state

                with tf.variable_scope('decode'):
                    WordRNN_output = tf.layers.dropout(WordRNN_output, rate=self.config.dropout_rate, training=self.is_training, name='drop_d')
                    logits = tf.layers.dense(WordRNN_output, units=self.config.vocabulary_size, activation=None, use_bias=True, name='fc_d')
                    predict = tf.argmax(logits, 1)
                    predicts.append(predict)
                    last_sentence.append(predict)

                tf.get_variable_scope().reuse_variables()
                if self.is_training:
                    WordRNN_last_word = self.sentences[:, sent_id*self.config.max_sentence_length + word_id]
                else:
                    WordRNN_last_word = predict

                # compute cross entropy loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sentences[:, sent_id*self.config.max_sentence_length + word_id], logits=logits)
                masked_cross_entropy = cross_entropy * self.masks[:, sent_id*self.config.max_sentence_length + word_id]
                cross_entropies.append(masked_cross_entropy)

                # compute acc
                ground_truth = tf.cast(self.sentences[:, sent_id*self.config.max_sentence_length + word_id], tf.int64)
                correct = tf.where(
                    tf.equal(predict, ground_truth),
                    tf.cast(self.masks[:, sent_id*self.config.max_sentence_length + word_id], tf.float32),
                    tf.cast(tf.zeros_like(predict), tf.float32)
                )
                corrects.append(correct)

        # 3. generate next ot last sentence
        for sent_id in range(1, self.config.max_sentence_num):
            # 3.1 get sentence feature
            with tf.variable_scope('word_embedding'):
                if self.is_training:
                    word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, self.sentences[:, (sent_id-1)*self.config.max_sentence_length : sent_id*self.config.max_sentence_length])
                else:
                    batch_sentences = tf.stack(last_sentence, axis=0)   # last_sentence shape = [max_sentence_length, batch_size]
                    batch_sentences_tran = tf.transpose(batch_sentences)
                    word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, batch_sentences_tran)

            self.semantic_features = self.sentence_encode(word_embeddings)

            # 3.2 init Word RNN
            with tf.variable_scope('word_rnn_initialize_%s' % sent_id, reuse=tf.AUTO_REUSE):
                # vis_context = tf.layers.dense(self.visual_feats, units=1024, activation=tf.tanh, use_bias=True, name='fc_v')
                context = tf.concat([self.visual_feats, self.semantic_features], axis=1)
                context = tf.layers.dropout(context, rate=self.config.dropout_rate, training=self.is_training, name='drop_s')
                init_c = tf.layers.dense(context, units=self.config.rnn_units, activation=tf.tanh, use_bias=True, name='fc_c')
                init_h = tf.layers.dense(context, units=self.config.rnn_units, activation=tf.tanh, use_bias=True, name='fc_h')

                WordRNN_last_state = init_c, init_h
                WordRNN_last_word = tf.zeros([self.batch_size], tf.int32)

            # 3.3 generate word one by one
            last_sentence = []
            for word_id in range(self.config.max_sentence_length):
                with tf.variable_scope("word_embedding"):
                    word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, WordRNN_last_word)

                with tf.variable_scope('word_rnn'):
                    WordRNN_output, WordRNN_state = WordRNN(word_embedding, WordRNN_last_state)
                    WordRNN_last_state = WordRNN_state

                with tf.variable_scope('decode'):
                    WordRNN_output = tf.layers.dropout(WordRNN_output, rate=self.config.dropout_rate, training=self.is_training, name='drop_d')
                    logits = tf.layers.dense(WordRNN_output, units=self.config.vocabulary_size, activation=None, use_bias=True, name='fc_d')
                    predict = tf.argmax(logits, 1)
                    predicts.append(predict)
                    last_sentence.append(predict)

                tf.get_variable_scope().reuse_variables()
                if self.is_training:
                    WordRNN_last_word = self.sentences[:, sent_id * self.config.max_sentence_length + word_id]
                else:
                    WordRNN_last_word = predict

                # compute cross entropy loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.sentences[:, sent_id * self.config.max_sentence_length + word_id],
                    logits=logits)
                masked_cross_entropy = cross_entropy * self.masks[:,
                                                       sent_id * self.config.max_sentence_length + word_id]
                cross_entropies.append(masked_cross_entropy)

                # compute acc
                ground_truth = tf.cast(self.sentences[:, sent_id * self.config.max_sentence_length + word_id],
                                       tf.int64)
                correct = tf.where(
                    tf.equal(predict, ground_truth),
                    tf.cast(self.masks[:, sent_id * self.config.max_sentence_length + word_id], tf.float32),
                    tf.cast(tf.zeros_like(predict), tf.float32)
                )
                corrects.append(correct)

        self.predicts = predicts
        self.cross_entropies = cross_entropies
        self.corrects = corrects

        print('rnn built.')

    def build_metrics(self):
        corrects = tf.stack(self.corrects, axis=1)
        self.accuracy = tf.reduce_sum(corrects) / tf.reduce_sum(self.masks)

        cross_entropies = tf.stack(self.cross_entropies, axis=1)
        self.cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(self.masks)

        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.cross_entropy_loss + self.reg_loss

        print('metrics built.')

    def build_optimizer(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.constant(self.config.learning_rate)
        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=self.config.decay_iters,
                decay_rate=self.config.decay_rate,
                staircase=True
            )

        learning_rate_decay_fn = _learning_rate_decay_fn
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            )

            self.step_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=self.global_step,
                learning_rate=learning_rate,
                optimizer=optimizer,
                clip_gradients=5.0,
                learning_rate_decay_fn=learning_rate_decay_fn,
                # variables=other_var_list
            )
        print('optimizer built.')

    def build_summary(self):
        with tf.name_scope("metrics"):
            tf.summary.scalar('cross entropy loss', self.cross_entropy_loss)
            tf.summary.scalar('reg loss', self.reg_loss)
            tf.summary.scalar('acc', self.accuracy)

        self.summary = tf.summary.merge_all()
        print('summary built.')

    def sentence_encode(self, word_embeddings):
        with tf.variable_scope('sentence_encode', reuse=tf.AUTO_REUSE):
            net = tf.layers.conv1d(word_embeddings, filters=1024, kernel_size=3, strides=1)
            sent_feature1 = tf.layers.max_pooling1d(net, pool_size=self.config.max_sentence_length - 2, strides=100)
            net = tf.layers.conv1d(net, filters=1024, kernel_size=3, strides=1)
            sent_feature2 = tf.layers.max_pooling1d(net, pool_size=self.config.max_sentence_length - 2 - 4, strides=100)
            net = tf.layers.conv1d(net, filters=1024, kernel_size=3, strides=1)
            sent_feature3 = tf.layers.max_pooling1d(net, pool_size=self.config.max_sentence_length - 2 - 6, strides=100)
        sent_feature1 = tf.reshape(sent_feature1, shape=[self.batch_size, 1024])
        sent_feature2 = tf.reshape(sent_feature2, shape=[self.batch_size, 1024])
        sent_feature3 = tf.reshape(sent_feature3, shape=[self.batch_size, 1024])
        semantic_features = tf.concat([sent_feature1, sent_feature2, sent_feature3], axis=1)
        return semantic_features