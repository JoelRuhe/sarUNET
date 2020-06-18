from networks.ops import *


def contracting_block(x, out_channels, dataformat, activation, param=None, is_training=True):

    if dataformat == "NCDHW":
        shape = x.get_shape().as_list()[2:5]
    else:
        shape = x.get_shape().as_list()[1:4]

    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv1_contract"):
        con1 = conv3d(x, out_channels, kernel, dataformat, activation, param)
        con1 = act(con1, activation, param)

    with tf.variable_scope("conv2_contract"):
        if dataformat == "NCDHW":
            con2 = conv3d(con1, con1.shape[1].value, kernel, dataformat, activation, param)
        else:
            con2 = conv3d(con1, con1.shape[4].value, kernel, dataformat, activation, param)

        con2 = act(con2, activation, param)

    return con2


def bottleneck(x, out_channels, dataformat, activation, param=None):
    """TODO: Implement with BatchNorm2d"""

    if dataformat == "NCDHW":
        shape = x.get_shape().as_list()[2:5]
    else:
        shape = x.get_shape().as_list()[1:4]

    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv1_bottle"):
        con1 = conv3d(x, out_channels, kernel, dataformat, activation, param)
        con1 = act(con1, activation, param)

    with tf.variable_scope("conv2_bottle"):
        con2 = conv3d(con1, out_channels, kernel, dataformat, activation, param)
        con2 = act(con2, activation, param)

    with tf.variable_scope("conv_trans_bottle"):
        trans = conv3d_transpose(con2, out_channels // 2, kernel, dataformat, activation, param)
        trans = act(trans, activation, param)

    return trans


def expansion_block(x, mid_channels, out_channels, dataformat, activation, param=None,
                    is_training=True):
    """TODO: Implement with BatchNorm2d"""

    if dataformat == "NCDHW":
        shape = x.get_shape().as_list()[2:5]
    else:
        shape = x.get_shape().as_list()[1:4]

    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv1_expanse"):
        con1 = conv3d(x, mid_channels, kernel, dataformat, activation, param)
        con1 = act(con1, activation, param)

    with tf.variable_scope("conv2_expanse"):
        con2 = conv3d(con1, mid_channels, kernel, dataformat, activation, param)
        con2 = act(con2, activation, param)

    with tf.variable_scope("conv_trans"):

        trans = conv3d_transpose(con2, out_channels, kernel, dataformat, activation, param)
        trans = act(trans, activation, param)

    return trans


def final_layer(x, mid_channels, out_channels, dataformat, activation, param=None):

    if dataformat == "NCDHW":
        shape = x.get_shape().as_list()[2:5]
    else:
        shape = x.get_shape().as_list()[1:4]

    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv1_final"):
        con1 = conv3d(x, mid_channels, kernel, dataformat, activation, param)
        con1 = act(con1, activation, param)

    with tf.variable_scope("conv2_final"):
        con2 = conv3d(con1, mid_channels, kernel, dataformat, activation, param)
        con2 = act(con2, activation, param)

    with tf.variable_scope("conv3_final"):
        con3 = conv3d(con2, out_channels, kernel, dataformat, activation, param)
        con3 = act(con3, activation, param)

    return con3


def crop_and_concat(upsampled, bypass, dataformat, crop=False):
    print(upsampled.shape, 'upsampled')
    print(bypass.shape, 'bypass')

    if crop:
        if dataformat == "NCDHW":
            if bypass.shape[4].value == 155:
                bypass = tf.pad(bypass, ([0,0],[0,0],[0,0],[0,0],[1,0]))
                print(bypass.shape, 'bypass2')
            d = (bypass.shape[2].value - upsampled.shape[2].value) // 2
            h = (bypass.shape[3].value - upsampled.shape[3].value) // 2
            w = (bypass.shape[4].value - upsampled.shape[4].value) // 2
            bypass = tf.pad(bypass, ([0, 0], [0, 0], [-d, -d], [-h, -h], [-w, -w]), "CONSTANT")
            return tf.concat([upsampled, bypass], 1)
        else:
            d = (bypass.shape[1].value - upsampled.shape[1].value) // 2
            h = (bypass.shape[2].value - upsampled.shape[2].value) // 2
            w = (bypass.shape[3].value - upsampled.shape[3].value) // 2
            bypass = tf.pad(bypass, ([0, 0], [-d, -d], [-h, -h], [-w, -w], [0, 0]), "CONSTANT")
            return tf.concat([upsampled, bypass], 4)

