import argparse
from networks.ops import *
from dataset import NumpyPathDataset
from tensorflow.data.experimental import AUTOTUNE
import os
import time
from networks.blocks import contracting_block, bottleneck, crop_and_concat, expansion_block, final_layer
from tqdm import tqdm
from utils import image_grid, uniform_box_sampler, parse_tuple
import horovod.tensorflow as hvd
import random
import numpy as np


def forward(x, args):
    tensor_list = list()

    # determine how many convolutions has to be done based on image size
    num_downsamples = int(np.log2(args.image_size) - np.log2(args.lowest_size))

    # CONTRACTION

    for i in range(num_downsamples):
        with tf.variable_scope("contract" + str(i)):
            x = contracting_block(x, args.first_output_channels, activation=args.activation, param=args.leakiness)
            #add tensor to list for later usage in Expanse
            tensor_list.append(x)
            x = maxpool3d(x, (2, 2, 2), (2, 2, 2), padding="SAME", data_format=args.data_format)
            args.first_output_channels = x.shape[1].value * 2

    # BOTTLENECK

    x = bottleneck(x, x.shape[1].value * 2, activation=args.activation, param=args.leakiness)

    # EXPANSE

    for i in reversed(range(len(tensor_list) - 1)):
        x = crop_and_concat(x, tensor_list[i + 1], crop=True)
        with tf.variable_scope("expanse" + str(i)):
            x = expansion_block(x, x.shape[1].value // 2, x.shape[1].value // 4, activation=args.activation,
                                param=args.leakiness)

    # FINAL

    x = crop_and_concat(x, tensor_list[0], crop=True)

    with tf.variable_scope("final"):
        x = final_layer(x, x.shape[1].value // 2, args.final_output_channels, activation=args.activation,
                        param=args.leakiness)

    return x


def main(args, config):
    if args.horovod:
        verbose = hvd.rank() == 0
        local_rank = hvd.local_rank()
    else:
        verbose = True
        local_rank = 0

    global_batch_size = args.batch_size * hvd.size() if args.horovod else args.batch_size

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtests', timestamp)

    global_step = 0

    tf.reset_default_graph()

    # ------------------------------------------------------------------------------------------#
    # DATASET

    data_path = os.path.join(args.dataset_root, f'{args.image_size}x{args.image_size}/')

    # retrieve dataset
    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0, is_correct_phase=True)

    # dataset = tf.data.Dataset.from_generator(npy_data.__iter__, npy_data.dtype, npy_data.shape)
    dataset = tf.data.Dataset.from_tensor_slices(npy_data.scratch_files)

    if args.horovod:
        dataset.shard(hvd.size(), hvd.rank())

    current_shape = [args.batch_size, args.image_channels, args.image_size // 4, args.image_size, args.image_size]
    real_image_input = tf.placeholder(shape=current_shape, dtype=tf.float32)

    # ------------------ NOISE ----------------

    rand_batch1 = np.random.rand(*real_image_input.shape) * 0.5
    noise_black_patches1 = rand_batch1.copy()

    # x_input = image_input + tf.random.normal(shape=image_input.shape) * args.noise_strength
    # x_input = image_input + tf.random.gamma(shape=x_input.shape, alpha=0.05)
    # x_input = x_input + tf.random.uniform(shape=x_input.shape) * args.noise_strength
    # x_input = x_input + tf.random.poisson(lam=0.5, shape=x_input.shape)

    #add box_sampler noise which mimics conebeam noise
    for i in range(real_image_input.shape[0]):
        for _ in range(200):
            arr_slices = uniform_box_sampler(noise_black_patches1, min_width=(1, 1, 2, 4, 4),
                                             max_width=(1, 1, 4, 8, 8))[0]
            noise_black_patches1[arr_slices] = 0

    x_input = real_image_input + noise_black_patches1
    y = real_image_input

    # ------------------ NETWORK ----------------

    prediction = forward(x_input, args)

    # ------------------ OPTIM -----------------
    if args.loss_fn is "mean_squared_error":
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    else:
        assert args.loss_fn != "mean_squared_error", "Choose one of the available args.loss_fn"

    lr_scaler = hvd.size() if args.horovod else 1
    optimizer = tf.train.AdamOptimizer(args.learning_rate * lr_scaler)

    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer)

    train_step = optimizer.minimize(loss)

    # ------------- SUMMARIES -------------
    train_input = tf.transpose(x_input[0], (1, 2, 3, 0))
    prediction_input = tf.transpose(prediction[0], (1, 2, 3, 0))
    real_input = tf.transpose(y[0], (1, 2, 3, 0))

    prediction_input = tf.clip_by_value(prediction_input, clip_value_min=args.clip_value_min,
                                        clip_value_max=args.clip_value_max)

    #transform images into grid
    shape = train_input.get_shape().as_list()
    grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
    grid_rows = shape[0] // grid_cols
    grid_shape = [grid_rows, grid_cols]
    train_input = image_grid(train_input, grid_shape, image_shape=shape[1:3],
                             num_channels=shape[-1])

    shape = prediction_input.get_shape().as_list()
    grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
    grid_rows = shape[0] // grid_cols
    grid_shape = [grid_rows, grid_cols]
    prediction_input = image_grid(prediction_input, grid_shape, image_shape=shape[1:3],
                                  num_channels=shape[-1])

    shape = real_input.get_shape().as_list()
    grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
    grid_rows = shape[0] // grid_cols
    grid_shape = [grid_rows, grid_cols]
    real_input = image_grid(real_input, grid_shape, image_shape=shape[1:3],
                            num_channels=shape[-1])

    with tf.variable_scope("train_summaries"):
        train_loss = tf.summary.scalar('train_loss', loss)
        train_imageNoise = tf.summary.image('train_imageNoise', train_input)
        train_imageRemake = tf.summary.image('train_imageRemake', prediction_input)
        train_imageReal = tf.summary.image('train_imageReal', real_input)

        image_summary_train = tf.summary.merge([train_loss, train_imageReal, train_imageRemake, train_imageNoise])

    with tf.variable_scope("test_summaries"):
        test_loss = tf.summary.scalar('test_loss', loss)
        test_imageNoise = tf.summary.image('test_imageNoise', train_input)
        test_imageRemake = tf.summary.image('test_imageRemake', prediction_input)
        test_imageReal = tf.summary.image('test_imageReal', real_input)

        image_summary_test = tf.summary.merge([test_loss, test_imageNoise, test_imageRemake, test_imageReal])

    # -------------- SESSION -------------

    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())

        if verbose:
            writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        #calculate percentage testset and trainingset
        train_size = int(len(npy_data) * args.train_size)
        test_size = int(len(npy_data) * (1 - args.train_size) + 1)

        num_train_steps = train_size // global_batch_size
        num_test_steps = test_size // global_batch_size

        for epoch in range(args.epochs1):
            epoch_loss_train = 0
            epoch_loss_test = 0

            # TRAINING
            for i in range(num_train_steps):

                #prepare trainingbatch
                batch_loc = np.random.randint(num_test_steps, len(npy_data) - args.batch_size)
                batch_paths = npy_data[batch_loc: batch_loc + args.batch_size]
                batch = np.stack(np.load(path) for path in batch_paths)
                batch = batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={real_image_input: batch})

                if i % args.logging_interval == 0 and verbose:
                    global_step = (epoch * num_train_steps * global_batch_size) + i * global_batch_size
                    writer.add_summary(summary, global_step)
                    writer.flush()
                    epoch_loss_train += c

            # TESTING
            for i in range(num_test_steps):

                #prepare testbatch
                batch_loc = np.random.randint(0, num_test_steps - args.batch_size)
                batch_paths = npy_data[batch_loc: batch_loc + args.batch_size]
                batch = np.stack(np.load(path) for path in batch_paths)
                batch = batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                c = sess.run(loss, feed_dict={real_image_input: batch})

                if i % args.logging_interval == 0 and verbose:
                    epoch_loss_test += c

            if verbose:
                # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss_test', simple_value=epoch_loss_test / num_test_steps)]), global_step)
                test_image_summary = sess.run(image_summary_test, feed_dict={real_image_input: batch})
                writer.add_summary(test_image_summary, global_step)
                writer.flush()

            if verbose:
                print(f'Epoch [{epoch}/{args.epochs}]\t'
                      f'Train Loss: {epoch_loss_train / num_train_steps}\t'
                      f'Test Loss: {epoch_loss_test / num_test_steps}\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, required=True, help="'height or width, eg: 128'")
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--scratch_path', type=str, required=True)
    parser.add_argument('--data_format', type=str, default='NCDHW')
    parser.add_argument('--noise_strength', type=float, default=0.003)
    parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--epochs', default=100000, type=int)
    parser.add_argument('--image_channels', default=1, type=int)
    parser.add_argument('--final_output_channels', default=1, type=int)
    parser.add_argument('--first_output_channels', default=64, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--lowest_size', default=8, type=int)
    parser.add_argument('--logging_interval', default=8, type=int)
    parser.add_argument('--clip_value_min', default=-1, type=int)
    parser.add_argument('--clip_value_max', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_size', default=0.9, type=float, help="percentage of the size of training set, "
                                                                      "eg: 0.9 (90%)")
    args = parser.parse_args()

    print('------------------ RUN CONFIRURATION --------------------\n')
    print('KEY\t\t\tVALUE')
    for arg in vars(args):
        print(f'{arg:<20}\t{getattr(args, arg):<40}')
    print('---------------------------------------------------------\n')

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



