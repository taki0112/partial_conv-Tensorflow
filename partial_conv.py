import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


##################################################################################
# Layer
##################################################################################
def partial_conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', scope='conv_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('mask'):
            _, h, w, _ = x.get_shape().as_list()

            slide_window = kernel * kernel
            mask = tf.ones(shape=[1, h, w, 1])

            update_mask = tf.layers.conv2d(mask, filters=1,
                                           kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                           strides=stride, padding=padding, use_bias=False, trainable=False)

            mask_ratio = slide_window / (update_mask + 1e-8)
            update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
            mask_ratio = mask_ratio * update_mask

        with tf.variable_scope('x'):
            x = tf.layers.conv2d(x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, padding=padding, use_bias=False)
            x = x * mask_ratio

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                x = tf.nn.bias_add(x, bias)
                x = x * update_mask

        return x
