from dataset import load_dataset
import tensorflow as tf
from model import unet
from utils import *
import random
import numpy as np
import os
import argparse
import horovod.tensorflow as hvd
import time


def main(args, config):
    if args.horovod:
        verbose = hvd.rank() == 0
    else:
        verbose = True

    global_batch_size = args.batch_size * hvd.size() if args.horovod else args.batch_size

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)

    if args.train:
        MRIdataset, SEGdataset = load_dataset(args, True)
    else:
        MRIdataset = load_dataset(args, False)


    if args.data_format == "NCDHW":
        current_shape = [args.batch_size, args.image_channels, args.image_size, args.image_size, args.image_size]
    else:
        current_shape = [args.batch_size, args.image_size, args.image_size, args.image_size, args.image_channels]


    MRI_image = tf.placeholder(shape=current_shape, dtype=tf.float32)

    if args.train:
        SEG_image = tf.placeholder(shape=current_shape, dtype=tf.float32)


    MRI_image = tf.ensure_shape(MRI_image, current_shape)
    if args.train:
        SEG_image = tf.ensure_shape(SEG_image, current_shape)

    prediction = unet(MRI_image, args)

    # Optimizer
    if args.train:
        loss = tf.losses.mean_squared_error(labels=SEG_image, predictions=prediction)

        lr_scaler = hvd.size() if args.horovod else 1
        optimizer = tf.train.AdamOptimizer(args.learning_rate * lr_scaler)

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer)

        train_step = optimizer.minimize(loss)

    # init_op = tf.initialize_all_variables()
    #
    # if args.train:
    #     saver = tf.train.Saver()

    # ------------- SUMMARIES -------------
    if args.data_format == "NCDHW":
        train_input = tf.transpose(MRI_image[0], (1, 2, 3, 0))
        prediction_input = tf.transpose(prediction[0], (1, 2, 3, 0))
        real_input = tf.transpose(SEG_image[0], (1, 2, 3, 0))
    else:
        train_input = tf.transpose(MRI_image[0], (0, 1, 2, 3))
        prediction_input = tf.transpose(prediction[0], (0, 1, 2, 3))
        real_input = tf.transpose(SEG_image[0], (0, 1, 2, 3))

    # prediction_input = tf.clip_by_value(prediction_input, clip_value_min=args.clip_value_min,
    #                                     clip_value_max=args.clip_value_max)

    # transform images into grid
    shape = train_input.get_shape().as_list()
    grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
    grid_rows = shape[0] // grid_cols
    grid_shape = [grid_rows, grid_cols]
    original_input = image_grid(train_input, grid_shape, image_shape=shape[1:3],
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
    segmentation_input = image_grid(real_input, grid_shape, image_shape=shape[1:3],
                            num_channels=shape[-1])

    with tf.variable_scope("train_summaries"):
        train_loss = tf.summary.scalar('train_loss', loss)
        train_original= tf.summary.image('train_original', original_input)
        train_prediction = tf.summary.image('train_prediction', prediction_input)
        train_segmentation = tf.summary.image('train_segmentation', segmentation_input)

        image_summary_train = tf.summary.merge([train_loss, train_original, train_prediction, train_segmentation])

    if args.train:
        with tf.variable_scope("test_summaries"):
            test_loss = tf.summary.scalar('test_loss', loss)
            test_original = tf.summary.image('test_original', original_input)
            test_prediction = tf.summary.image('test_prediction', prediction_input)
            test_segmentation = tf.summary.image('test_segmentation', segmentation_input)

            image_summary_test = tf.summary.merge([test_loss, test_original, test_prediction, test_segmentation])

    #Session

    with tf.Session(config=config) as sess:

        # if not args.train:
        #     saver.restore(sess, "tmp/test_model.ckpt")

        sess.run(tf.initialize_all_variables())

        if verbose:
            writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        #calculate percentage testset and trainingset
        train_size = int(len(MRIdataset) * args.train_size)
        test_size = int(len(MRIdataset) * (1 - args.train_size) + 1)

        num_train_steps = train_size // global_batch_size
        num_test_steps = test_size // global_batch_size

        for epoch in range(args.epochs):
            epoch_loss_train = 0
            epoch_loss_test = 0

            # Training (and testing of args.train is False)
            for i in range(num_train_steps):
                # prepare trainingbatch
                if args.train:
                    batch_loc = np.random.randint(num_test_steps, len(MRIdataset) - args.batch_size)

                    SEG_paths = SEGdataset[batch_loc: batch_loc + args.batch_size]
                    SEGbatch = np.stack(np.load(k, allow_pickle=True) for k in SEG_paths)

                    MRI_paths = MRIdataset[batch_loc: batch_loc + args.batch_size]
                    MRIbatch = np.stack(np.load(path, allow_pickle=True) for path in MRI_paths)

                    SEGbatch = np.transpose(SEGbatch, (0, 4, 1, 2, 3))
                    MRIbatch = np.transpose(MRIbatch, (0, 4, 1, 2, 3))

                    # print('training...')
                    # print(SEG_paths)
                    # print(MRI_paths)

                    if args.data_format == "NDHWC":
                        MRIbatch = np.transpose(MRIbatch, (0, 1, 2, 3, 4))
                        SEGbatch = np.transpose(SEGbatch, (0, 1, 2, 3, 4))

                    _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={MRI_image: MRIbatch, SEG_image: SEGbatch})

                else:
                    batch_loc = np.random.randint(num_test_steps, len(MRIdataset) - args.batch_size)

                    MRI_paths = MRIdataset[batch_loc: batch_loc + args.batch_size]
                    MRIbatch = np.stack(np.load(path, allow_pickle=True) for path in MRI_paths)

                    MRIbatch = np.transpose(MRIbatch, (0, 4, 1, 2, 3))

                    # print('Final testing')

                    if args.data_format == "NDHWC":
                        MRIbatch = np.transpose(MRIbatch, (0, 1, 2, 3, 4))

                    _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={MRI_image: MRIbatch})

                if verbose:
                    global_step = (epoch * num_train_steps * global_batch_size) + i * global_batch_size
                    writer.add_summary(summary, global_step)
                    writer.flush()
                    epoch_loss_train += c

            # TESTING
            if args.train:
                for i in range(num_test_steps):

                    # prepare testbatch
                    batch_loc = np.random.randint(0, num_test_steps - args.batch_size)

                    SEG_paths = SEGdataset[batch_loc: batch_loc + args.batch_size]
                    SEGbatch = np.stack(np.load(k, allow_pickle=True) for k in SEG_paths)

                    MRI_paths = MRIdataset[batch_loc: batch_loc + args.batch_size]
                    MRIbatch = np.stack(np.load(path, allow_pickle=True) for path in MRI_paths)

                    SEGbatch = np.transpose(SEGbatch, (0, 4, 1, 2, 3))
                    MRIbatch = np.transpose(MRIbatch, (0, 4, 1, 2, 3))

                    # print('testing...')
                    # print(SEG_paths)
                    # print(MRI_paths)

                    if args.data_format == "NDHWC":
                        MRIbatch = np.transpose(MRIbatch, (0, 1, 2, 3, 4))
                        SEGbatch = np.transpose(SEGbatch, (0, 1, 2, 3, 4))

                    c = sess.run(loss, feed_dict={MRI_image: MRIbatch, SEG_image: SEGbatch})

                    if i % args.logging_interval == 0 and verbose:
                        epoch_loss_test += c

                if verbose:
                    # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss_test', simple_value=epoch_loss_test / num_test_steps)]), global_step)
                    test_image_summary = sess.run(image_summary_test, feed_dict={MRI_image: MRIbatch, SEG_image: SEGbatch})
                    writer.add_summary(test_image_summary, global_step)
                    writer.flush()

                # saver.save(sess, "tmp/test_model.ckpt")

            if verbose:
                print(f'Epoch [{epoch}/{args.epochs}]\t'
                      f'Train Loss: {epoch_loss_train / num_train_steps}\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=128, type=int)
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
    parser.add_argument('--lowest_size', default=16, type=int)
    parser.add_argument('--logging_interval', default=8, type=int)
    parser.add_argument('--clip_value_min', default=-1, type=int)
    parser.add_argument('--clip_value_max', default=2, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--image_type', default='T1c', choices=['T1', 'T1c', 'T2', 'Flair'], type=str)
    parser.add_argument('--image_tissue', default='HGG', choices=['HGG', 'LGG'], type=str)
    parser.add_argument('--num_copy', default=1, type=int, help="Amount of classes that are copied to scratch folder")
    parser.add_argument('--train_size', default=0.9, type=float, help="percentage of the size of training set, "
                                                                      "eg: 0.9 (90%)")

    args = parser.parse_args()

    print('------------------ RUN CONFIRURATION --------------------\n')
    print('KEY\t\t\tVALUE')
    for arg in vars(args):
        print(f'{arg:<20}\t{getattr(args, arg):<40}')
    print('---------------------------------------------------------\n')

    # assert float(np.log2(args.image_size)) == int(np.log2(args.image_size))

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

# # prepare trainingbatch
# # batch_loc = np.random.randint(len(dataset) - args.batch_size)
# batch_loc = np.random.randint(len(MRIdataset))
#
# SEG_paths = SEGdataset[batch_loc: batch_loc + args.batch_size]
# SEGbatch = np.stack(np.load(k, allow_pickle=True) for k in SEG_paths)
# # SEGbatch = SEGbatch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
#
# # MRI_paths = MRIdataset[batch_loc: batch_loc + args.batch_size]
# # MRIbatch = np.stack(np.load(path, allow_pickle=True) for path in MRI_paths)
# # MRIbatch = MRIbatch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
#
# MRI_paths = MRIdataset[batch_loc: batch_loc + args.batch_size]
# MRIbatch = np.stack(np.load(path, allow_pickle=True) for path in MRI_paths)
# # MRIbatch = MRIbatch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
#
#
# # SEGbatch = _remove_colormap(SEGbatch)
# SEGbatch = np.transpose(SEGbatch, (0, 4, 1, 2, 3))
# MRIbatch = np.transpose(MRIbatch, (0, 4, 1, 2, 3))
#
# if args.data_format == "NDHWC":
#     # MRIbatch = np.transpose(MRIbatch, (0, 2, 3, 4, 1))
#     SEGbatch = np.transpose(SEGbatch, (0, 1, 4, 3, 2))

# _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={MRI_image: MRIbatch, SEG_image: SEGbatch})
