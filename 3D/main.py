import argparse
from networks.ops import *
from dataset import imagenet_dataset
from tensorflow.data.experimental import AUTOTUNE
import os
import time
import h5py


def contracting_block(x, out_channels, activation, param=None, is_training=True):
    """TODO: Implement with BatchNorm2d"""
    print(out_channels)
    with tf.variable_scope("conv1_contract"):
        print()
        shape = x.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, out_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_contract"):
        print()
        shape = con1.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, con1.shape[1].value, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    return con2


def bottle_neck(x, out_channels, activation, param=None):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv1_bottle"):
        print()
        shape = x.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con1 = conv3d(x, out_channels, kernel, activation, param)
        con1 = act(con1, activation, param)
        print("Shape = " + str(con1.shape))

    with tf.variable_scope("conv2_bottle"):
        print()
        shape = con1.get_shape().as_list()[1:4]
        kernel = [k(s) for s in shape]
        con2 = conv3d(con1, out_channels, kernel, activation, param)
        con2 = act(con2, activation, param)
        print("Shape = " + str(con2.shape))

    with tf.variable_scope("conv_trans_bottle"):
        print()
        shape = con2.get_shape().as_list()[1:4]
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
        d = (bypass.shape[2].value - upsampled.shape[2].value) // 2
        h = (bypass.shape[3].value - upsampled.shape[3].value) // 2
        w = (bypass.shape[4].value - upsampled.shape[4].value) // 2
        bypass = tf.pad(bypass, ([0, 0], [0, 0], [-d, -d], [-h, -h], [-w, -w]), "CONSTANT")

    return tf.concat([upsampled, bypass], 1)


def forward(x, args):
    counter = 0
    tensor_list = list()

    while args.image_size > 4:
        args.image_size = args.image_size // 2
        counter = counter + 1

    print("\n-----CONTRACTION-----")

    for i in range(counter):
        print("contract" + str(i))
        with tf.variable_scope("contract" + str(i)):
            x = contracting_block(x, args.first_output_channels, activation=args.activation, param=args.leakiness)
            tensor_list.append(x)
            print("maxpool")
            x = maxpool3d(x, (2, 2, 2), (2, 2, 2), padding="SAME", data_format=args.data_format)
            args.first_output_channels = x.shape[1].value * 2

    print("\n-----BOTTLENECK-----")

    x = bottle_neck(x, x.shape[1].value * 2, activation=args.activation, param=args.leakiness)

    print("\n-----EXPANSION-----")

    for i in reversed(range(len(tensor_list) - 1)):
        x = crop_and_concat(x, tensor_list[i + 1], crop=True)
        print("expanse" + str(i))
        with tf.variable_scope("expanse" + str(i)):
            x = expansion_block(x, x.shape[1].value // 2, x.shape[1].value // 4, activation=args.activation,
                                param=args.leakiness)

    x = crop_and_concat(x, tensor_list[0], crop=True)
    print("\n-----FINAL-----")

    with tf.variable_scope("final"):
        x = final_layer(x, x.shape[1].value // 2, args.final_output_channels, activation=args.activation,
                        param=args.leakiness)

    return x


def main(args):
    is_train = tf.placeholder(tf.bool, name="condition")
    print(args.image_size)
    # dataset_train, imagenet_data_train = imagenet_dataset(args.dataset_path, args.scratch_path, args.image_size,
    #                                                       args.train, copy_files=True, num_labels=args.num_labels)
    # dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    # dataset_train = dataset_train.repeat()
    # dataset_train = dataset_train.prefetch(AUTOTUNE)
    # dataset_train = dataset_train.make_one_shot_iterator()
    #
    # dataset_test, imagenet_data_test = imagenet_dataset(args.dataset_path, args.scratch_path, args.image_size,
    #                                                     args.train, copy_files=True, num_labels=args.num_labels)
    # dataset_test = dataset_test.batch(args.batch_size, drop_remainder=True)
    # dataset_test = dataset_test.repeat()
    # dataset_test = dataset_test.prefetch(AUTOTUNE)
    # dataset_test = dataset_test.make_one_shot_iterator()

    with h5py.File("input/3d-mnist/full_dataset_vectors.h5", "r") as hf:
        X_train = hf["X_train"][:]
        y_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        y_test = hf["y_test"][:]

    batch = tf.cond(is_train, lambda: X_train//100, lambda: X_test//100)

    # if len(batch) == 1:
    #     image_input = batch
    #     label = None
    # elif len(batch) == 2:
    #     image_input, label = batch
    # else:
    #     raise NotImplementedError()
    image_input = tf.reshape(batch, [args.batch_size, 3, 32, 32, 32])

    print(X_train.shape)
    print(image_input.shape)

    image_input = tf.ensure_shape(image_input, [args.batch_size, args.image_channels, args.image_size, args.image_size, args.image_size])
    image_input = tf.cast(image_input, tf.float32)
    x_input = image_input + tf.random.normal(shape=image_input.shape) * 0.15
    y = image_input

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    prediction = forward(x_input, args)
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    x_input = tf.transpose(x_input, perm=[0, 2, 3, 4, 1])
    y = tf.transpose(y, perm=[0, 2, 3, 4, 1])
    prediction = tf.transpose(prediction, perm=[0, 2, 3, 4, 1])

    prediction = tf.clip_by_value(prediction, clip_value_min=0, clip_value_max=1)

    with tf.variable_scope("train_summaries"):
        train_loss = tf.summary.scalar('loss', loss)
        train_imageNoise = tf.summary.image('train_imageNoise', x_input)
        train_imageRemake = tf.summary.image('train_imageRemake', prediction)
        train_imageReal = tf.summary.image('train_imageReal', y)

    with tf.variable_scope("test_summaries"):
        test_imageNoise = tf.summary.image('test_imageNoise', x_input)
        test_imageRemake = tf.summary.image('test_imageRemake', prediction)
        test_imageReal = tf.summary.image('test_imageReal', y)

    image_summary_train = tf.summary.merge([train_loss, train_imageReal, train_imageRemake, train_imageNoise])
    image_summary_test = tf.summary.merge([test_imageNoise, test_imageRemake, test_imageReal])

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())

        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        for epoch in range(args.epochs):
            epoch_loss_train = 0
            epoch_loss_test = 0
            for i in range(len(X_train) // args.batch_size):
                train = True
                _, summary, c = sess.run([optimizer, image_summary_train, loss], feed_dict={is_train: train})
                global_step = (epoch * (
                    len(X_train)) // args.batch_size * args.batch_size) + i * args.batch_size
                writer.add_summary(summary, global_step)

                epoch_loss_train = epoch_loss_train + c
                writer.flush()

            for i in range(len(X_test) // args.batch_size):
                train = False
                c = sess.run(loss, feed_dict={is_train: train})

                epoch_loss_test = epoch_loss_test + c

            writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag='loss_test', simple_value=epoch_loss_test / args.testset_length)]), global_step)

            test_image_summary = sess.run(image_summary_test, feed_dict={is_train: train})
            writer.add_summary(test_image_summary, global_step)

            writer.flush()

            print('Epoch', epoch, 'completed out of', args.epochs, 'loss_train:',
                  epoch_loss_train / args.trainset_length,
                  'loss_test:',
                  epoch_loss_test / args.testset_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_size', type=int, help="'height or width, eg: 128'")
    parser.add_argument('batch_size', type=int, default=100)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset_path', type=str, default='/nfs/managed_datasets/imagenet-full/')
    parser.add_argument('--scratch_path', type=str, default='/scratch/joelr/')
    parser.add_argument('--data_format', type=str, default='NCDHW')
    parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--image_channels', default=3, type=int)
    parser.add_argument('--final_output_channels', default=3, type=int)
    parser.add_argument('--first_output_channels', default=64, type=int)
    parser.add_argument('--testset_length', default=50, type=int)
    parser.add_argument('--trainset_length', default=1000, type=int)

    args = parser.parse_args()

    main(args)

