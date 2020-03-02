import argparse
from networks.ops import *
from dataset import NumpyPathDataset
from dataset import npy_data
from tensorflow.data.experimental import AUTOTUNE
import os
import time
from networks.blocks import contracting_block, bottleneck, crop_and_concat, expansion_block, final_layer
from tqdm import tqdm
from utils import image_grid
import horovod.tensorflow as hvd
import random




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

    x = bottleneck(x, x.shape[1].value * 2, activation=args.activation, param=args.leakiness)

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


def main(args, config):

    verbose = hvd.rank() == 0 if args.horovod else True
    global_batch_size = args.batch_size * hvd.size() if args.horovod else args.batch_size

    print(verbose)
    print(args.horovod)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)

    is_train = tf.placeholder(tf.bool, name="condition")

    def load(x):
        x = np.load(x.decode())[np.newaxis, ...].astype(np.float32) / 1024 - 1
        return x

    data_path = os.path.join(args.dataset_root, f'{args.image_size}x{args.image_size}/')

    dataset_train, npy_data_train = npy_data(data_path, args.scratch_path, train_size=args.train_size, train=True, copy_files=True, is_correct_phase=True)

    dataset_train = dataset_train.map(lambda x: tuple(tf.py_func(load, [x], [tf.float32])), num_parallel_calls=AUTOTUNE)
    dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.prefetch(AUTOTUNE)
    dataset_train = dataset_train.make_one_shot_iterator()

    dataset_test, npy_data_test = npy_data(data_path, args.scratch_path, train_size=args.train_size, train=False, copy_files=False, is_correct_phase=True)

    dataset_test = dataset_test.map(lambda x: tuple(tf.py_func(load, [x], [tf.float32])), num_parallel_calls=AUTOTUNE)
    dataset_test = dataset_test.batch(args.batch_size, drop_remainder=True)
    dataset_test = dataset_test.repeat()
    dataset_test = dataset_test.prefetch(AUTOTUNE)
    dataset_test = dataset_test.make_one_shot_iterator()

    image_input = tf.cond(is_train, lambda: dataset_train.get_next(), lambda: dataset_test.get_next())

    # if len(batch) == 1:
    #     image_input = batch
    #     label = None
    # elif len(batch) == 2:
    #     image_input, label = batch
    # else:
    #     raise NotImplementedError()

    # image_input = tf.squeeze(image_input, axis=0)

    image_input = tf.ensure_shape(image_input, [args.batch_size, args.image_channels, args.image_size//4, args.image_size, args.image_size])

    x_input = image_input + tf.random.normal(shape=image_input.shape) * args.noise_strength
    x_input = image_input + tf.random.gamma(shape=x_input.shape, alpha=0.03)
    x_input = x_input + tf.random.uniform(shape=x_input.shape) * args.noise_strength
    x_input = x_input + tf.random.poisson(lam=0.1, shape=x_input.shape)

    y = image_input

    # ------------------ NETWORK ----------------
    prediction = forward(x_input, args)

    # ------------------ OPTIM -----------------
    if args.loss_fn is "mean_squared_error":
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    lr_scaler = hvd.size() if args.horovod else 1
    optimizer = tf.train.AdamOptimizer(args.learning_rate * lr_scaler)

    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer)

    train_step = optimizer.minimize(loss)

    # ------------- SUMMARIES -------------
    train_input = tf.transpose(x_input[0], (1, 2, 3, 0))
    prediction_input = tf.transpose(prediction[0], (1, 2, 3, 0))
    real_input = tf.transpose(y[0], (1, 2, 3, 0))

    # prediction_input = tf.clip_by_value(prediction_input, clip_value_min=0, clip_value_max=1)

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
        train_loss = tf.summary.scalar('loss', loss)

        train_imageNoise = tf.summary.image('train_imageNoise', train_input)
        train_imageRemake = tf.summary.image('train_imageRemake', prediction_input)
        train_imageReal = tf.summary.image('train_imageReal', real_input)
        image_summary_train = tf.summary.merge([train_loss, train_imageReal, train_imageRemake, train_imageNoise])

    with tf.variable_scope("test_summaries"):
        test_imageNoise = tf.summary.image('test_imageNoise', train_input)
        test_imageRemake = tf.summary.image('test_imageRemake', prediction_input)
        test_imageReal = tf.summary.image('test_imageReal', real_input)
        image_summary_test = tf.summary.merge([test_imageNoise, test_imageRemake, test_imageReal])

    # -------------- SESSION -------------

    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())

        if verbose:
            writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        for epoch in range(args.epochs):
            epoch_loss_train = 0
            epoch_loss_test = 0
            num_train_steps = len(npy_data_train) // global_batch_size
            num_test_steps = len(npy_data_test) // global_batch_size

            train = True
            for i in tqdm(range(num_train_steps)):
                _, summary, c = sess.run([train_step, image_summary_train, loss], feed_dict={is_train: train})

                if i % args.logging_interval == 0 and verbose:
                    global_step = (epoch * num_train_steps * global_batch_size) + i * global_batch_size
                    writer.add_summary(summary, global_step)
                    writer.flush()
                    epoch_loss_train += c

            train = False
            for i in tqdm(range(num_test_steps)):
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
    parser.add_argument('--image_size', type=int, help="'height or width, eg: 128'")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--scratch_path', type=str, default='/scratch/joelrnew/')
    parser.add_argument('--data_format', type=str, default='NCDHW')
    parser.add_argument('--noise_strength', type=float, default=0.003)
    parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--image_channels', default=1, type=int)
    parser.add_argument('--final_output_channels', default=1, type=int)
    parser.add_argument('--first_output_channels', default=64, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--logging_interval', default=8, type=int)
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



