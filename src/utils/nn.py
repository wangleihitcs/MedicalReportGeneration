import tensorflow as tf
import tensorflow.contrib.layers as layers

def kernel_initializer():
    return tf.random_uniform_initializer(minval = -0.08, maxval = 0.08)

def kernel_regularizer():
    return layers.l2_regularizer(scale=1e-4)

def dense(inputs, units, activation=tf.tanh, use_bias=True, name=None):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        use_bias=use_bias,
        trainable=True,
        kernel_initializer=kernel_initializer(),
        kernel_regularizer=kernel_regularizer(),
        activity_regularizer=None,
        name=name)

def dropout(inputs, rate, is_training, name):
    return tf.layers.dropout(inputs, rate=rate, training=is_training, name=name)