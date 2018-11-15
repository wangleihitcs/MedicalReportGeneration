import tensorflow as tf
from utils import nn
from nets import inception


class Model(object):
    def __init__(self, is_training=True):
        self.batch_size = 32
        self.vocabulary_size = 1966     # 1963 word + '<S>' '</S>' '<EOS>'

        self.lstm_units = 512
        self.embedding_size = 512
        self.initial_learning_rate = 1e-4
        self.image_size = 224
        self.decay_rate = 0.9
        self.decay_epochs = 5 * (2876 / self.batch_size)

        self.max_sentence_num = 6
        self.max_caption_length = 50
        self.max_caption = 300

        self.lstm_drop_rate = 0.3
        self.dropout_rate = 0.5
        self.is_train = is_training

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.sentences = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_caption])
        self.masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_caption])

        self.build_cnn()
        self.build_rnn()
        self.build_metrics()
        if is_training:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        net_local, _, net_global = inception.inception_v3(self.images, trainable=True, is_training=True, add_summaries=False)

        # net_local = tf.reshape(net_local, [self.batch_size, 25, 2048])
        # self.visual_feats = tf.reduce_max(net_local, axis=1)

        self.visual_feats = net_global # net_global=[batch size, 2048]
        print('cnn built.')

    def build_rnn(self):
        with tf.variable_scope("word_embedding"):
            word_embedding_matrix = tf.get_variable(
                                    name='weights',
                                    shape=[self.vocabulary_size, self.embedding_size],
                                    initializer=nn.kernel_initializer(),
                                    regularizer=nn.kernel_regularizer(),
                                    trainable=True)

        # 1. build Word LSTM
        WordLSTM = tf.nn.rnn_cell.LSTMCell(
            self.lstm_units,
            initializer=nn.kernel_initializer())
        if self.is_train:
            WordLSTM = tf.nn.rnn_cell.DropoutWrapper(
                WordLSTM,
                input_keep_prob=1.0 - self.lstm_drop_rate,
                output_keep_prob=1.0 - self.lstm_drop_rate,
                state_keep_prob=1.0 - self.lstm_drop_rate)


        predictions = []  # store predict word
        prediction_corrects = []  # store correct predict to compute accuracy
        cross_entropies = []  # store cross entropy loss
        # 3. generate word step by step
        for sent_id in range(1):
            with tf.variable_scope("word_lstm_initialize"):
                context = self.visual_feats
                initial_memory = nn.dense(context, self.lstm_units, name='fc_a')
                initial_output = nn.dense(context, self.lstm_units, name='fc_b')
            WordLSTM_last_state = initial_memory, initial_output
            WordLSTM_last_word = tf.zeros([self.batch_size], tf.int32)  # tf.zeros() means the '<S>' token

            for id in range(self.max_caption_length):
                with tf.variable_scope("word_embedding"):
                    word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, WordLSTM_last_word)

                with tf.variable_scope('WordLSTM'):
                    inputs = tf.concat([word_embedding], axis=1)
                    WordLSTM_current_output, WordLSTM_current_state = WordLSTM(inputs, WordLSTM_last_state)

                with tf.variable_scope('decode'):
                    expanded_output = nn.dropout(WordLSTM_current_output, self.dropout_rate, self.is_train, name='drop')
                    logits = nn.dense(expanded_output, units=self.vocabulary_size, activation=None, name='fc')
                    prediction = tf.argmax(logits, 1)
                    predictions.append(prediction)
                tf.get_variable_scope().reuse_variables()

                WordLSTM_last_state = WordLSTM_current_state
                # use teacher policy
                if self.is_train:
                    WordLSTM_last_word = self.sentences[:, sent_id*self.max_caption_length + id]
                else:
                    WordLSTM_last_word = prediction

                # compute loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.sentences[:, sent_id*self.max_caption_length + id],
                                logits=logits)
                masked_cross_entropy = cross_entropy * self.masks[:, sent_id*self.max_caption_length + id]
                cross_entropies.append(masked_cross_entropy)

                # compute accuracy
                ground_truth = tf.cast(self.sentences[:, sent_id*self.max_caption_length + id], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(self.masks[:, sent_id*self.max_caption_length + id], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32)
                )
                prediction_corrects.append(prediction_correct)

        for sent_id in range(1, self.max_sentence_num):
            with tf.variable_scope('sent_encode'):
                word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix, self.sentences[:, (sent_id-1)*self.max_caption_length:sent_id*self.max_caption_length])
                sent_feats = self.sentence_encode(word_embeddings)

            with tf.variable_scope("word_lstm_initialize2", reuse=tf.AUTO_REUSE):
                sent_feats = nn.dropout(sent_feats, self.dropout_rate, self.is_train, name='drop')
                # context = tf.concat([self.visual_feats, sent_feats], axis=1)
                context = sent_feats
                initial_memory = nn.dense(context, self.lstm_units, name='fc_a')
                initial_output = nn.dense(context, self.lstm_units, name='fc_b')
            WordLSTM_last_state = initial_memory, initial_output
            WordLSTM_last_word = tf.zeros([self.batch_size], tf.int32)  # tf.zeros() means the '<S>' token

            for id in range(self.max_caption_length):
                with tf.variable_scope("word_embedding"):
                    word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, WordLSTM_last_word)

                with tf.variable_scope('WordLSTM'):
                    inputs = tf.concat([word_embedding], axis=1)
                    WordLSTM_current_output, WordLSTM_current_state = WordLSTM(inputs, WordLSTM_last_state)

                with tf.variable_scope('decode'):
                    expanded_output = nn.dropout(WordLSTM_current_output, self.dropout_rate, self.is_train, name='drop')
                    logits = nn.dense(expanded_output, units=self.vocabulary_size, activation=None, name='fc')
                    prediction = tf.argmax(logits, 1)
                    predictions.append(prediction)
                # tf.get_variable_scope().reuse_variables()

                WordLSTM_last_state = WordLSTM_current_state
                # use teacher policy
                if self.is_train:
                    WordLSTM_last_word = self.sentences[:, sent_id * self.max_caption_length + id]
                else:
                    WordLSTM_last_word = prediction

                # compute loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.sentences[:, sent_id * self.max_caption_length + id],
                    logits=logits)
                masked_cross_entropy = cross_entropy * self.masks[:, sent_id * self.max_caption_length + id]
                cross_entropies.append(masked_cross_entropy)

                # compute accuracy
                ground_truth = tf.cast(self.sentences[:, sent_id * self.max_caption_length + id], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(self.masks[:, sent_id * self.max_caption_length + id], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32)
                )
                prediction_corrects.append(prediction_correct)

        # 4. compute accuracy
        prediction_corrects = tf.stack(prediction_corrects, axis=1)
        accuracy = tf.reduce_sum(prediction_corrects) / tf.reduce_sum(self.masks)

        self.predictions = predictions
        self.cross_entropies = cross_entropies
        self.accuracy = accuracy
        print('rnn built.')

    def build_metrics(self):
        cross_entropies = tf.stack(self.cross_entropies, axis=1)
        self.cross_entropy_text = tf.reduce_sum(cross_entropies) / tf.reduce_sum(self.masks)

        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.cross_entropy_text + self.reg_loss

        print('metrics built.')

    def build_optimizer(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.constant(self.initial_learning_rate)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=self.decay_epochs,
                decay_rate=self.decay_rate,
                staircase=True
            )

        learning_rate_decay_fn = _learning_rate_decay_fn
        # learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.GradientDescentOptimizer(self.initial_learning_rate)
            # optimizer = tf.train.MomentumOptimizer(
            #     learning_rate = tf.constant(self.initial_learning_rate),
            #     momentum = 0.9,
            #     use_nesterov = True
            # )

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
                learning_rate_decay_fn=learning_rate_decay_fn
            )
        print('optimizer built.')

    def build_summary(self):
        with tf.name_scope("metrics"):
            tf.summary.scalar('cross entropy', self.cross_entropy_text)
            tf.summary.scalar('reg loss', self.reg_loss)
            tf.summary.scalar('acc', self.accuracy)

        self.summary = tf.summary.merge_all()
        print('summary built.')

    def sentence_encode(self, word_embeddings):
        # word_embeddings shape = [batch_size, max_sentence_length, 512]
        with tf.variable_scope('sent_decode', reuse=tf.AUTO_REUSE):
            net = tf.layers.conv1d(word_embeddings, filters=1024, kernel_size=3, strides=1)
            sent_feature1 = tf.layers.max_pooling1d(net, pool_size=self.max_caption_length - 2, strides=100)
            net = tf.layers.conv1d(net, filters=1024, kernel_size=3, strides=1)
            sent_feature2 = tf.layers.max_pooling1d(net, pool_size=self.max_caption_length - 4, strides=100)
            net = tf.layers.conv1d(net, filters=1024, kernel_size=3, strides=1)
            sent_feature3 = tf.layers.max_pooling1d(net, pool_size=self.max_caption_length - 6, strides=100)
        sent_feature1 = tf.reshape(sent_feature1, shape=[self.batch_size, 1024])
        sent_feature2 = tf.reshape(sent_feature2, shape=[self.batch_size, 1024])
        sent_feature3 = tf.reshape(sent_feature3, shape=[self.batch_size, 1024])
        sent_feature = tf.concat([sent_feature1, sent_feature2, sent_feature3], axis=1)  # [batch_size, 1024*3]
        # print(sent_feature)
        return sent_feature
