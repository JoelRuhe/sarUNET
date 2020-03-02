import argparse
from networks.ops import *
from dataset import imagenet_dataset
from tensorflow.data.experimental import AUTOTUNE
import horovod.tensorflow as hvd
import os
import time
import random
import numpy as np
from networks.blocks import contracting_block, bottleneck, crop_and_concat, expansion_block, final_layer


def forward(x, args):

    image_channels = x.shape[1]
    # Collect nodes that need to be connected further downstream.
    tensor_list = list()

    image_size = x.get_shape().as_list()[-1]
    # Number of downsample steps to go down to a spatial resolution of 4x4.
    num_downsamples = int(np.log2(image_size) - np.log2(4))

    for i in range(num_downsamples):
        with tf.variable_scope("contract_" + str(i)):
            num_channels = args.base_channels if i == 0 else x.shape[1] * 2
            x = contracting_block(x, num_channels, activation=args.activation, param=args.leakiness)
            tensor_list.append(x)
            # TODO: Replace with bilinear up/down
            x = maxpool2d(x, (2, 2), (2, 2))

    x = bottleneck(x, x.shape[1] * 2, activation=args.activation, param=args.leakiness)

    for i in reversed(range(len(tensor_list) - 1)):
        x = crop_and_concat(x, tensor_list[i + 1], crop=True)
        with tf.variable_scope("expand_" + str(i)):
            # TODO: Replace with bilinear up/down
            x = expansion_block(x, x.shape[1] // 2, x.shape[1] // 4, activation=args.activation, param=args.leakiness)

    x = crop_and_concat(x, tensor_list[0], crop=True)

    with tf.variable_scope("expand_" + str(len(tensor_list))):
        x = final_layer(x, x.shape[1].value // 2, image_channels, activation=args.activation, param=args.leakiness)

    return x


def main(args, config):

    verbose = hvd.rank() == 0 if args.horovod else True
    global_batch_size = args.batch_size * hvd.size() if args.horovod else args.batch_size

    # ---------------- DATASET ---------------

    # Train data pipeline
    is_train = tf.placeholder(tf.bool, name="is_train")
    dataset_train, imagenet_data_train = imagenet_dataset(args.dataset_root, args.scratch_path, args.image_size,
                                                          train=True, copy_files=verbose, num_labels=args.num_labels)
    dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.prefetch(AUTOTUNE)
    dataset_train = dataset_train.make_one_shot_iterator()

    # Test data pipeline
    dataset_test, imagenet_data_test = imagenet_dataset(args.dataset_root, args.scratch_path, args.image_size,
                                                        train=False, copy_files=verbose, num_labels=args.num_labels)
    dataset_test = dataset_test.batch(args.batch_size, drop_remainder=True)
    dataset_test = dataset_test.repeat()
    dataset_test = dataset_test.prefetch(AUTOTUNE)
    dataset_test = dataset_test.make_one_shot_iterator()

    # Fetch batch conditioned on the mode (train/test) we're in.
    batch = tf.cond(is_train, lambda: dataset_train.get_next(), lambda: dataset_test.get_next())

    if len(batch) == 1:
        image_input = batch
        label = None
    elif len(batch) == 2:
        image_input, label = batch
    else:
        raise NotImplementedError()

    image_channels = image_input.shape[1]
    image_input = tf.ensure_shape(image_input, [args.batch_size, image_channels, args.image_size, args.image_size])

    x_input = image_input + tf.random.normal(shape=image_input.shape) * args.noise_strength
    x_input = x_input + tf.random.uniform(shape=x_input.shape) * args.noise_strength


    y = image_input

    # ------------------ NETWORK ----------------

    prediction = forward(x_input, args)

    # ------------------ OPTIM -----------------

    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    lr_scaler = hvd.size() if args.horovod else 1
    optimizer = tf.compat.v1.train.AdamOptimizer(args.learning_rate * lr_scaler)

    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer)

    train_step = optimizer.minimize(loss)

    # ------------- SUMMARIES -------------

    # Transpose to valid image format.
    x_input = tf.transpose(x_input, perm=[0, 2, 3, 1])
    y = tf.transpose(y, perm=[0, 2, 3, 1])
    prediction = tf.transpose(prediction, perm=[0, 2, 3, 1])

    # Clip to obtain valid image values.

    prediction = tf.clip_by_value(prediction, clip_value_min=0, clip_value_max=1)
    with tf.variable_scope("train_summaries"):
        train_loss = tf.summary.scalar('loss', loss)
        train_x = tf.summary.image('train_x', x_input)
        train_pred = tf.summary.image('train_pred', prediction)
        train_y = tf.summary.image('train_y', y)
        image_summary_train = tf.summary.merge([train_loss, train_y, train_pred, train_x])

    with tf.variable_scope("test_summaries"):
        test_x = tf.summary.image('test_x', x_input)
        test_pred = tf.summary.image('test_pred', prediction)
        test_y = tf.summary.image('test_y', y)
        image_summary_test = tf.summary.merge([test_x, test_pred, test_y])

    # -------------- SESSION -------------

    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())
        if verbose:
            timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
            logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)
            writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        for epoch in range(args.epochs):
            epoch_loss_train = 0
            epoch_loss_test = 0
            num_train_steps = len(imagenet_data_train) // global_batch_size
            num_test_steps = len(imagenet_data_test) // global_batch_size
            train = True
            for i in range(num_train_steps):
                _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={is_train: train})

                if i % args.logging_interval == 0 and verbose:
                    global_step = (epoch * num_train_steps * global_batch_size) + i * global_batch_size
                    writer.add_summary(summary, global_step)
                    writer.flush()
                    epoch_loss_train += c

            train = False
            for i in range(num_test_steps):
                c = sess.run(loss, feed_dict={is_train: train})
            if verbose:
                epoch_loss_test += c
                writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag='loss_test', simple_value=epoch_loss_test / num_test_steps)]), global_step)
                test_image_summary = sess.run(image_summary_test, feed_dict={is_train: train})
                writer.add_summary(test_image_summary, global_step)
                writer.flush()

            if verbose:
                print(f'Epoch [{epoch}/{args.epochs}]\t'
                      f'Train Loss: {epoch_loss_train / num_train_steps}\t'
                      f'Test Loss: {epoch_loss_test / num_test_steps}\t')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['imagenet'])
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, help="'height or width, eg: 128'", required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--scratch_path', type=str, default=f'/scratch/{os.environ["USER"]}/')
    parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
    parser.add_argument('--noise_strength', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--base_channels', default=64, type=int, help='Controls network complexity (parameters).')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--logging_interval', default=8, type=int)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    print('------------------ RUN CONFIRURATION --------------------\n')
    print('KEY\t\t\tVALUE')
    for arg in vars(args):
        print(f'{arg:<20}\t{getattr(args, arg):<40}')
    print('---------------------------------------------------------\n')

    # Assert image_size is a multiple of two.
    assert float(np.log2(args.image_size)) == int(np.log2(args.image_size))

    if args.horovod:
        hvd.init()
        np.random.seed(args.seed + hvd.rank())
        tf.random.set_random_seed(args.seed + hvd.rank())
        random.seed(args.seed + hvd.rank())

        print(f"Rank {hvd.rank()}:{hvd.local_rank()} reporting!")

    else:
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)
        random.seed(args.seed)

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)

    if args.gpu:
        config.gpu_options.allow_growth = True
        if args.horovod:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    else:
        config = tf.ConfigProto(graph_options=gopts,
                                intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                                inter_op_parallelism_threads=args.num_inter_ops,
                                allow_soft_placement=True,
                                device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

    main(args, config)
