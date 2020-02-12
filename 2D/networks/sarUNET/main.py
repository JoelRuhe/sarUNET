from networks.ops import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

BATCH_SIZE = 100
n_classes = 10
sample_shape = [3, 16, 16]

x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None, n_classes])


def contracting_block(x, out_channels, scope1, scope2, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_contract"):
        print()
        print(scope1 + ":")
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, out_channels, kernel,  activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_contract"):
        print()
        print(scope2 + ":")
        shape = con1.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, con1.shape[1].value*2, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    return con2


def bottle_neck(x, out_channels, activation, param=None):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_bottle"):
        print()
        print("conv1 :")
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, out_channels, kernel,  activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_bottle"):
        print()
        print("conv2 :")
        shape = con1.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, out_channels, kernel,  activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans_bottle"):
        print()
        print("conv_trans")
        shape = con2.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        trans = conv2d_transpose(con2, out_channels//2, kernel, activation, param)
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans


def expansion_block(x, mid_channels, out_channels, conv1, conv2, conv_trans, stop, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_expanse"):
        print()
        print(conv1 + " :")
        shape = x.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_expanse"):
        print()
        print(conv2 + " :")
        shape = con1.get_shape().as_list()[2:]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    if not stop:
        with tf.variable_scope("conv_trans"):
            print()
            print(conv_trans + " :")
            shape = con2.get_shape().as_list()[2:]
            kernel = [k(s) for s in shape]
            trans = conv2d_transpose(con2, out_channels, kernel, activation, param)
            trans = act(trans, activation, param)
            print("Shape = " + str(trans.shape))

        return trans
    return con2


def forward(x):
    output_shape=32

    print("\n-----CONTRACTING-----")

    with tf.variable_scope("contract1"):
        x1 = contracting_block(x, output_shape, "conv1", "conv2", activation='leaky_relu', param=0.2)
        print("maxpool")
        p1 = maxpool2d(x1, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    with tf.variable_scope("contract2"):
        x2 = contracting_block(p1, p1.shape[1].value * 2, "conv3", "conv4", activation='leaky_relu', param=0.2)
        print("maxpool")
        p2 = maxpool2d(x2, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    with tf.variable_scope("contract3"):
        x3 = contracting_block(p2, p2.shape[1].value * 2, "conv5", "conv6", activation='leaky_relu', param=0.2)
        print("maxpool")
        p3 = maxpool2d(x3, (2, 2), (2, 2), padding="SAME", data_format="NCHW")

    print("\n-----BOTTLENECK-----")

    bottle = bottle_neck(p3, p3.shape[1].value * 2, activation='leaky_relu', param=0.2)

    print("\n-----EXPANSION-----")

    with tf.variable_scope("expanse3"):
        x4 = expansion_block(bottle, p3.shape[1].value // 2, p3.shape[1].value // 4, "conv1", "conv2", "conv_trans3", False,
                             activation='leaky_relu', param=0.2)

    with tf.variable_scope("expanse2"):
        x5 = expansion_block(x4, x4.shape[1].value // 2, x4.shape[1].value // 4, "conv4", "conv5", "conv_trans6", False,
                             activation='leaky_relu', param=0.2)

    with tf.variable_scope("expanse1"):
        x6 = expansion_block(x5, x5.shape[1].value // 2, x5.shape[1].value // 4, "conv7", "conv8", "conv_trans9", True,
                             activation='leaky_relu', param=0.2)

    return x6


if __name__=="__main__":
    # forward(x)
    # x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE] + sample_shape)
    # print(x_input.shape)
    x = tf.reshape(x, shape=[100, 28, 28, 1])
    print("test")
    prediction = forward(x)
    print("TEEEEESTSTSTS")
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    output_shape = 32

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


