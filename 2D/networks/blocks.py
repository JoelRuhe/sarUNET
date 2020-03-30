from networks.ops import *


def contracting_block(x, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""
    with tf.variable_scope("conv_1"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, out_channels, kernel, activation, param)
        x = act(x, activation, param)
        # x = horovod_batch_normalization(x)

    with tf.variable_scope("conv_2"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, x.shape[1].value, kernel, activation, param)
        x = act(x, activation, param)

    return x


def bottleneck(x, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, out_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_2"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, out_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_trans"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d_transpose(x, out_channels // 2, kernel, activation, param)
        x = act(x, activation, param)

    return x


def expansion_block(x, mid_channels, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, mid_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_2"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, mid_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_trans"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d_transpose(x, out_channels, kernel, activation, param)
        x = act(x, activation, param)

    return x


def final_layer(x, mid_channels, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, mid_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_2"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, mid_channels, kernel, activation, param)
        x = act(x, activation, param)

    with tf.variable_scope("conv_3"):
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        x = conv2d(x, out_channels, kernel, activation, param)
        x = act(x, activation, param)

    return x


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        h = (bypass.shape[2].value - upsampled.shape[2].value) // 2
        w = (bypass.shape[3].value - upsampled.shape[3].value) // 2
        bypass = tf.pad(bypass, ([0, 0], [0, 0], [-h, -h], [-w, -w]), "CONSTANT")

    return tf.concat([upsampled, bypass], 1)
