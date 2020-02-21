import tensorflow as tf
import numpy as np
# import horovod.tensorflow as hvd


def k(x):
    """Don't use kernels smaller than actual size."""
    if x < 3:
        return 1
    else:
        return 3


def calculate_gain(activation, param=None):
    """Calculate by how much weights should be adjusted to retain sufficient input size."""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if activation in linear_fns or activation == 'sigmoid':
        return 1
    elif activation == 'tanh':
        return 5.0 / 3
    elif activation == 'relu':
        return np.sqrt(2.0)
    elif activation == 'leaky_relu':
        assert param is not None
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))

    elif activation == 'leaky_relu_native':
        assert param is not None
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))

    else:
        raise ValueError("Unsupported nonlinearity {}".format(activation))


def get_weight(shape, activation, lrmul=1, use_eq_lr=False, param=None):
    """Get a weight variable."""
    fan_in = np.prod(shape[:-1])
    gain = calculate_gain(activation, param)
    he_std = gain / np.sqrt(fan_in)
    if use_eq_lr:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    w = tf.get_variable("weight", shape=shape,
                            initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef

    if use_eq_lr:
        w *= runtime_coef

    return w


def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[3]], initializer=tf.initializers.random_normal()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])


def dense(x, fmaps, activation, lrmul=1, param=None):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], activation, lrmul=lrmul, param=param)
    return tf.matmul(x, w)


def conv2d(x, fmaps, kernel, activation, param=None, lrmul=1):
    print("Kernel = " + str(kernel))
    w = get_weight([*kernel, x.shape[1].value, fmaps], activation, param=param, lrmul=lrmul)
    print("Weight = " + str(w.shape))
    print(x.shape)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


def conv2d_transpose(x, fmaps, kernel, activation, param=None, lrmul=1):
    print("Kernel = " + str(kernel))
    print(fmaps)
    print("x.shape = " + str(x.shape))
    output_shape = tf.stack([x.shape[0].value, fmaps, int(x.shape[2].value*2), int(x.shape[2].value*2)])
    print("output_shape = " + str(output_shape.shape))
    w = get_weight([*kernel, x.shape[1].value, fmaps], activation, param=param, lrmul=lrmul)
    print("Weight1 = " + str(w.shape))
    w = tf.transpose(w, perm=[0, 1, 3, 2])
    print("Weight2 = " + str(w.shape))
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[2, 2], padding='SAME', data_format='NCHW')


def maxpool2d(x, pool_size, strides, padding, data_format):
    return tf.nn.max_pool2d(x, pool_size, strides, padding, data_format)


def conv3d(x, fmaps, kernel, activation, param=None, lrmul=1):
    w = get_weight([*kernel, x.shape[1].value, fmaps], activation, param=param, lrmul=lrmul)
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')


def leaky_relu(x, alpha_lr=0.2):
    with tf.variable_scope('leaky_relu'):
        return tf.nn.leaky_relu(x, alpha=alpha_lr)


def act(x, activation, param=None):
    if activation == 'leaky_relu':
        assert param is not None
        return leaky_relu(x, alpha_lr=param)
    elif activation == 'linear':
        return x
    elif activation == 'leaky_relu_native':
        return tf.nn.leaky_relu(x)
    else:
        raise ValueError(f"Unknown activation {activation}")


def horovod_batch_normalization(x, is_training=True, decay=.9, data_format='channels_first'):

    shape = [1 for _ in range(len(x.shape))]
    if data_format == 'channels_first':
        shape[1] = x.shape[1]
        mean, var = tf.nn.moments(x, keepdims=True, axes=[0] + list(range(2, len(shape))))

    elif data_format == 'channels_last':
        shape[-1] = x.shape[-1]
        mean, var = tf.nn.moments(x, keepdims=True, axes=list(range(0, len(shape[:-1]))))

    else:
        raise ValueError(f"Unknown data format {data_format}.")

    gamma = tf.get_variable('gamma', shape=shape, initializer=tf.initializers.ones())
    beta = tf.get_variable('beta', shape=shape, initializer=tf.initializers.zeros())

    # global_mean = hvd.allreduce(mean)
    # global_var = hvd.allreduce(var)

    ema_mean = tf.get_variable('ema_mean', shape=mean.shape, initializer=tf.initializers.zeros(), trainable=False)
    ema_var = tf.get_variable('ema_var', shape=var.shape, initializer=tf.initializers.ones(), trainable=False)

    # ema_mean = decay * ema_mean + (1 - decay) * global_mean
    # ema_var = decay * ema_var + (1 - decay) * global_var

    if not is_training:
        global_mean = ema_mean
        global_var = ema_var

    return tf.nn.batch_normalization(x, global_mean, global_var, offset=beta, scale=gamma, variance_epsilon=1e-8)


