from networks.ops import *


def contracting_block(x, out_channels, activation, param=None, is_training=True):

    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]

    x = conv3d(out_channels, kernel, activation, param)
    x = act(x, activation, param)
    x = horovod_batch_normalization(x, is_training=is_training)

    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]
    x = conv3d(out_channels, kernel, activation, param)
    x = act(x, activation, param)
    x = horovod_batch_normalization(x, is_training)


