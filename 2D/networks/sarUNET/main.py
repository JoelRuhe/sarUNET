from networks.ops import *
BATCH_SIZE = 100
sample_shape = [3, 16, 16]


def contracting_block(x, out_channels, scope1, scope2, activation, param=None, is_training=True):

    print(scope1 + ":")
    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]
    with tf.variable_scope(scope1):
        con1 = conv2d(x, out_channels, kernel, activation, param)
        con1 = act(con1, activation, param)

    print(scope2 + ":")
    shape = con1.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]
    with tf.variable_scope(scope2):
        con2 = conv2d(con1, out_channels*2, kernel, activation, param)
        con2 = act(con2, activation, param)

    return con2


def bottle_neck(x):
    """TODO: Implement Bottleneck"""
    horovod_batch_normalization(x)


def expansion_block(x, out_channels, activation, param=None, is_training=True):
    print("Conv6:")
    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv6"):
        con6 = conv2d_transpose(x, out_channels, kernel, activation, param)
        print(con6.shape)
        con6 = act(con6, activation, param)

    print("Conv6:")
    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv6"):
        con6 = conv2d_transpose(x, out_channels, kernel, activation, param)
        print(con6.shape)
        con6 = act(con6, activation, param)

    print("Conv6:")
    shape = x.get_shape().as_list()[2:]
    kernel = [k(s) for s in shape]

    with tf.variable_scope("conv6"):
        con6 = conv2d_transpose(x, out_channels, kernel, activation, param)
        print(con6.shape)
        con6 = act(con6, activation, param)


if __name__=="__main__":
    x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE] + sample_shape)
    print(x_input.shape)
    x1 = contracting_block(x_input, x_input.shape[1].value, "conv1", "conv2", activation='leaky_relu', param=0.2)
    p1 = maxpool2d(x1, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    x2 = contracting_block(p1, x_input.shape[1].value, "conv3", "conv4", activation='leaky_relu', param=0.2)
    p2 = maxpool2d(x2, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    x3 = contracting_block(p2, x_input.shape[1].value, "conv5", "conv6", activation='leaky_relu', param=0.2)
    p3 = maxpool2d(x3, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    # bottle_neck(x)
    x = expansion_block(p3, int(p3.shape[1].value/2), activation='leaky_relu', param=0.2)
