import tensorflow as tf

a = tf.constant(1.0)

with tf.Session() as sess:
    print(sess.run(a))