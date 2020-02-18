from networks.opscopy import *
from tensorflow.examples.tutorials.mnist import input_data
import os
import time

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
BATCH_SIZE = 100
n_classes = 10
activation = 'leaky_relu'

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 32, 32, 1])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 32, 32, 1])



gopts = tf.GraphOptions(place_pruned_graph=True)
config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)

timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)
writer = tf.summary.FileWriter(logdir=logdir)

print("Arguments passed:")
print(f"Saving files to {logdir}")

var_list = list()


def contracting_block(x, out_channels, scope1, scope2, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_contract"):
        print()
        print(scope1 + ":")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, out_channels, kernel, activation, param)
        # con1 = tf.Print(con1, [tf.norm(con1), tf.norm(y)], "Conv1 before act: ")
        con1 = act(con1, activation, param)
        # con1 = tf.Print(con1, [tf.norm(con1), tf.norm(y)], "Conv1 after act: ")
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_contract"):
        print()
        print(scope2 + ":")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, con1.shape[3].value, kernel, activation, param)
        # con2 = tf.Print(con2, [tf.norm(con2), tf.norm(y)], "Conv2 before act: ")
        con2 = act(con2, activation, param)
        # con2 = tf.Print(con2, [tf.norm(con2), tf.norm(y)], "Conv2 after act: ")
        print("Shape = " + str(con2.shape))

    return con2


def bottle_neck(x, out_channels, activation, param=None):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_bottle"):
        print()
        print("conv1 :")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, out_channels, kernel,  activation, param)
        con1 = act(con1, activation, param)

        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_bottle"):
        print()
        print("conv2 :")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, out_channels, kernel,  activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans_bottle"):
        print()
        print("conv_trans")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        print(con2.shape)
        print(out_channels//2)
        trans = conv2d_transpose(x, out_channels//2, kernel, activation, param)
        # trans = tf.transpose(trans, perm=(0,2,3,1))
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans


def expansion_block(x, mid_channels, out_channels, conv1, conv2, conv_trans, stop, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_expanse"):
        print()
        print(conv1 + " :")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]

        print(x.shape)
        print(mid_channels)
        print(out_channels)
        con1 = conv2d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_expanse"):
        print()
        print(conv2 + " :")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans"):
        print()
        print(conv_trans + " :")
        shape = con2.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        trans = conv2d_transpose(con2, out_channels, kernel, activation, param)
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans


def final_layer(x, mid_channels, out_channels, activation, param=None, is_training=True):
    with tf.variable_scope("conv1_final"):
        print()
        print("conv1 :")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))
        # con1 = tf.Print(con1, [tf.norm(con1), tf.norm(y)])

    with tf.variable_scope("conv2_final"):
        print()
        print("conv2 :")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))
        # con2 = tf.Print(con2, [tf.norm(con2), tf.norm(y)])

    with tf.variable_scope("conv3_final"):
        print()
        print("conv3 :")
        shape = con2.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con3 = conv2d(con2, out_channels, kernel, activation, param)
        con3 = act(con3, activation, param)
        print("Shape = " + str(con3.shape))

    return con3


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        print(upsampled.shape)
        print(bypass.shape)
        h = (bypass.shape[1].value - upsampled.shape[1].value) // 2
        w = (bypass.shape[2].value - upsampled.shape[2].value) // 2
        print(h)
        print(w)
        bypass = tf.pad(bypass, ([0, 0], [-h, -h], [-w, -w], [0, 0]), "CONSTANT")

        print("padding succeeded")
        print(upsampled.shape)
        print(bypass.shape)

    return tf.concat([upsampled, bypass], 3)


def forward(x):
    output_shape=64
    print("\n-----CONTRACTING-----")
    x = tf.reshape(x, shape=[-1, 32, 32, 1])
    # x = tf.Print(x, [tf.norm(x), tf.norm(y)], "START: ")
    with tf.variable_scope("contract1"):
        x1 = contracting_block(x, output_shape, "conv1", "conv2", activation=activation, param=0.2)
        print("maxpool")
        p1 = maxpool2d(x1, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        # p1 = tf.Print(p1, [tf.norm(p1), tf.norm(y)], "Pool1: ")

    with tf.variable_scope("contract2"):
        x2 = contracting_block(p1, p1.shape[3].value * 2, "conv3", "conv4", activation=activation, param=0.2)
        print("maxpool")
        p2 = maxpool2d(x2, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        # p2 = tf.Print(p2, [tf.norm(p2), tf.norm(y)], "Pool2: ")

    with tf.variable_scope("contract3"):
        x3 = contracting_block(p2, p2.shape[3].value * 2, "conv5", "conv6", activation=activation, param=0.2)
        print("maxpool")
        p3 = maxpool2d(x3, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        # p3 = tf.Print(p3, [tf.norm(p3), tf.norm(y)])

    print("\n-----BOTTLENECK-----")

    bottle = bottle_neck(p3, p3.shape[3].value * 2, activation=activation, param=0.2)
    # bottle = tf.Print(bottle, [tf.norm(bottle), tf.norm(y)])

    print("\n-----EXPANSION-----")

    decode_block3 = crop_and_concat(bottle, x3, crop=True)

    with tf.variable_scope("expanse3"):
        x4 = expansion_block(decode_block3, decode_block3.shape[3].value // 2, decode_block3.shape[3].value // 4, "conv1", "conv2", "conv_trans3", False,
                             activation=activation, param=0.2)

    decode_block2 = crop_and_concat(x4, x2, crop=True)

    with tf.variable_scope("expanse2"):
        x5 = expansion_block(decode_block2, decode_block2.shape[3].value // 2, decode_block2.shape[3].value // 4, "conv4", "conv5", "conv_trans6", False,
                             activation=activation, param=0.2)

    decode_block1 = crop_and_concat(x5, x1, crop=True)

    print("\n-----FINAL-----")

    with tf.variable_scope("final"):
        x6 = final_layer(decode_block1, decode_block1.shape[3].value // 2, 1, activation=activation, param=0.2)

    tf.summary.scalar('loss', x6)

    return x6


if __name__=="__main__":

    prediction = forward(x)

    # prediction = tf.Print(prediction, [tf.norm(prediction)], "Prediction: ")
    # loss = tf.contrib.losses.sigmoid
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    summary_graph = tf.summary.scalar('loss', loss)
    summary_imageRemake = tf.summary.image('imageRemake', x)
    summary_imageReal = tf.summary.image('imageReal', y)

    # tf.summary.merge_all()


    # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad != None else (grad, var) for grad, var in optimizer]
    # train_step = optimizer.apply_gradients(capped_gvs)
    output_shape = 32

    hm_epochs = 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples // 10000)):
                print(int(mnist.train.num_examples / 1000))
                print(i)
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)

                epoch_x = np.reshape(epoch_x, [BATCH_SIZE, 28, 28, 1])
                epoch_y = np.reshape(epoch_x, [BATCH_SIZE, 28, 28, 1])

                epoch_x = np.pad(epoch_x, ((0,0), (2,2), (2,2), (0,0)))
                epoch_y = np.pad(epoch_y, ((0,0), (2,2), (2,2), (0,0)))

                _, summary_session, summary_imageReal1, summary_imageRemake1, c = sess.run([optimizer, summary_graph, summary_imageReal, summary_imageRemake, loss], feed_dict={x: epoch_x, y: epoch_y})
                writer.add_summary(summary_session, epoch)
                writer.add_summary(summary_imageReal1, epoch)
                writer.add_summary(summary_imageRemake1, epoch)

                # writer.flush()
                var_list = c
                print(c, 'loss')

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: epoch_x, y: epoch_y}))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))




