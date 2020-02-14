from networks.opscopy import *
from tensorflow.examples.tutorials.mnist import input_data
import cv2

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

BATCH_SIZE = 100
n_classes = 10

x = tf.placeholder(tf.float32, shape=[100, 32, 32, 1])
y = tf.placeholder(tf.float32, shape=[100, 32, 32, 1])



def contracting_block(x, out_channels, scope1, scope2, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_contract"):
        print()
        print(scope1 + ":")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, out_channels, kernel,  activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_contract"):
        print()
        print(scope2 + ":")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, con1.shape[3].value, kernel, activation, param)
        con2 = act(con2, activation, param)
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

    # if not stop:
    with tf.variable_scope("conv_trans"):
        print()
        print(conv_trans + " :")
        shape = con2.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        trans = conv2d_transpose(con2, out_channels, kernel, activation, param)
        # trans = tf.transpose(trans, perm=(0,2,3,1))
        trans = act(trans, activation, param)
        print("Shape = " + str(trans.shape))

    return trans
    # return con2


def final_layer(x, mid_channels, out_channels, activation, param=None, is_training=True):
    with tf.variable_scope("conv1_final"):
        print()
        print("conv1 :")
        shape = x.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con1 = conv2d(x, mid_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))
        con1 = tf.Print(con1, [tf.norm(con1), tf.norm(y)])

    with tf.variable_scope("conv2_final"):
        print()
        print("conv2 :")
        shape = con1.get_shape().as_list()[1:3]
        kernel = [k(s) for s in shape]
        con2 = conv2d(con1, mid_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))
        con2 = tf.Print(con2, [tf.norm(con2), tf.norm(y)])

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

    # def crop_and_concat(self, upsampled, bypass, crop=False):
    #     if crop:
    #         c = (bypass.size()[2] - upsampled.size()[2]) // 2
    #         bypass = F.pad(bypass, (-c, -c, -c, -c))
    #     return torch.cat((upsampled, bypass), 1)


def forward(x):
    output_shape=64
    print("\n-----CONTRACTING-----")
    x = tf.reshape(x, shape=[-1, 32, 32, 1])
    with tf.variable_scope("contract1"):
        x1 = contracting_block(x, output_shape, "conv1", "conv2", activation='leaky_relu', param=0.2)
        print("maxpool")
        p1 = maxpool2d(x1, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        p1 = tf.Print(p1, [tf.norm(p1), tf.norm(y)])

    with tf.variable_scope("contract2"):
        x2 = contracting_block(p1, p1.shape[3].value * 2, "conv3", "conv4", activation='leaky_relu', param=0.2)
        print("maxpool")
        p2 = maxpool2d(x2, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        p2 = tf.Print(p2, [tf.norm(p2), tf.norm(y)])

    with tf.variable_scope("contract3"):
        x3 = contracting_block(p2, p2.shape[3].value * 2, "conv5", "conv6", activation='leaky_relu', param=0.2)
        print("maxpool")
        p3 = maxpool2d(x3, (2, 2), (2, 2), padding="SAME", data_format="NHWC")
        p3 = tf.Print(p3, [tf.norm(p3), tf.norm(y)])

    print("\n-----BOTTLENECK-----")

    bottle = bottle_neck(p3, p3.shape[3].value * 2, activation='leaky_relu', param=0.2)
    bottle = tf.Print(bottle, [tf.norm(bottle), tf.norm(y)])

    print("\n-----EXPANSION-----")

    decode_block3 = crop_and_concat(bottle, x3, crop=True)

    with tf.variable_scope("expanse3"):
        x4 = expansion_block(decode_block3, decode_block3.shape[3].value // 2, decode_block3.shape[3].value // 4, "conv1", "conv2", "conv_trans3", False,
                             activation='leaky_relu', param=0.2)

    decode_block2 = crop_and_concat(x4, x2, crop=True)

    with tf.variable_scope("expanse2"):
        x5 = expansion_block(decode_block2, decode_block2.shape[3].value // 2, decode_block2.shape[3].value // 4, "conv4", "conv5", "conv_trans6", False,
                             activation='leaky_relu', param=0.2)

    decode_block1 = crop_and_concat(x5, x1, crop=True)

    print("\n-----FINAL-----")

    with tf.variable_scope("final"):
        x6 = final_layer(decode_block1, decode_block1.shape[3].value // 2, 1, activation='leaky_relu', param=0.2)

    return x6


if __name__=="__main__":

    prediction = forward(x)
    prediction = tf.Print(prediction, [tf.norm(prediction), tf.norm(y)])
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    optimizer = tf.train.GradientDescentOptimizer(0.0001, True).minimize(loss)

    output_shape = 32

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)

                epoch_x = np.reshape(epoch_x, [100, 28, 28, 1])
                epoch_y = np.reshape(epoch_x, [100, 28, 28, 1])

                epoch_x = np.pad(epoch_x, ((0,0), (2,2), (2,2), (0,0)))
                epoch_y = np.pad(epoch_y, ((0,0), (2,2), (2,2), (0,0)))

                epoch_x = epoch_x/255
                epoch_y = epoch_y/255

                _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
                epoch_x = tf.nn.l2_loss(epoch_x)
                epoch_loss += c
                print(c)

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


