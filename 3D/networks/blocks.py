from networks.ops import *


def contracting_block(x, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""
    print(out_channels)
    with tf.variable_scope("conv1_contract"):
        print()
        shape = x.get_shape().as_list()[2:5]
        print(shape)
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, out_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_contract"):
        print()
        shape = con1.get_shape().as_list()[2:5]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, con1.shape[1].value, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    return con2


def bottleneck(x, out_channels, activation, param=None):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_bottle"):
        print()
        shape = x.get_shape().as_list()[2:5]
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, out_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_bottle"):
        print()
        shape = con1.get_shape().as_list()[2:5]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, out_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans_bottle"):
        print()
        shape = con2.get_shape().as_list()[2:5]
        kernel = [k(s) for s in shape]
        trans = conv3d_transpose(con2, out_channels // 2, kernel, activation, param)
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans


def expansion_block(x, mid_channels, out_channels, activation, param=None,
                    is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_expanse"):
        print()
        shape = x.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_expanse"):
        print()
        shape = con1.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans"):
        print()
        shape = con2.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        trans = conv3d_transpose(con2, out_channels, kernel, activation, param)
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans


def final_layer(x, mid_channels, out_channels, activation, param=None):
    with tf.variable_scope("conv1_final"):
        print()
        print("conv1 :")
        shape = x.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_final"):
        print()
        print("conv2 :")
        shape = con1.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv3_final"):
        print()
        print("conv3 :")
        shape = con2.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con3 = conv3d(con2, out_channels, kernel, activation, param)
        con3 = act(con3, activation, param)
        print("Shape = " + str(con3.shape))

    return con3


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        print(upsampled.shape)
        print(bypass.shape)
        print(bypass.shape[1].value)
        print(upsampled.shape[1].value)

        d = (bypass.shape[2].value - upsampled.shape[2].value) // 2
        h = (bypass.shape[3].value - upsampled.shape[3].value) // 2
        w = (bypass.shape[4].value - upsampled.shape[4].value) // 2
        bypass = tf.pad(bypass, ([0, 0], [0, 0], [-d, -d], [-h, -h], [-w, -w]), "CONSTANT")

    return tf.concat([upsampled, bypass], 1)