import argparse
from networks.ops import *
from dataset import NumpyPathDataset
from tensorflow.data.experimental import AUTOTUNE
import os
import time
from networks.blocks import contracting_block, bottleneck, crop_and_concat, expansion_block, final_layer
from tqdm import tqdm
from utils import image_grid


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


def main(args):
    is_train = tf.placeholder(tf.bool, name="condition")

    def load(x):
        x = np.load(x.decode())[np.newaxis, ...].astype(np.float32) / 1024 - 1
        return x

    data_path = os.path.join(args.dataset_path, f'{args.image_size}x{args.image_size}/')

    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=True,
                                is_correct_phase=True)

    dataset = tf.data.Dataset.from_tensor_slices(npy_data.scratch_files)
    dataset = dataset.shuffle(len(npy_data))
    dataset = dataset.map(lambda x: tuple(tf.py_func(load, [x], [tf.float32])), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.make_one_shot_iterator()
    data = dataset.get_next()

    if len(data) == 1:
        image_input = data
        label = None
    elif len(data) == 2:
        image_input, label = data
    else:
        raise NotImplementedError()

    image_input = tf.squeeze(image_input, axis=0)
    image_input = tf.ensure_shape(image_input, [args.batch_size, args.image_channels, args.image_size//4, args.image_size, args.image_size])

    x_input = image_input + tf.random.normal(shape=image_input.shape) * 0.1
    x_input = tf.cast(x_input, tf.int32)
    x_input = x_input + tf.random.gamma(x_input, alpha=1)

    y = image_input

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    prediction = forward(x_input, args)
    loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    train_input = tf.transpose(x_input[0], (1, 2, 3, 0))
    prediction_input = tf.transpose(prediction[0], (1, 2, 3, 0))
    real_input = tf.transpose(y[0], (1, 2, 3, 0))

    prediction = tf.clip_by_value(prediction, clip_value_min=0, clip_value_max=1)

    with tf.variable_scope("train_summaries"):
        train_loss = tf.summary.scalar('loss', loss)
        train_imageNoise = tf.summary.image('train_imageNoise', train_input)
        train_imageRemake = tf.summary.image('train_imageRemake', prediction_input)
        train_imageReal = tf.summary.image('train_imageReal', real_input)

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

            for i in tqdm(range(1000)):
                train = True
                _, summary, c = sess.run([optimizer, image_summary_train, loss], feed_dict={is_train: train})
                global_step = (epoch * (
                    1000)) + i * args.batch_size
                writer.add_summary(summary, global_step)

                epoch_loss_train = epoch_loss_train + c
                writer.flush()

            print('Epoch', epoch, 'completed out of', args.epochs, 'loss_train:',
                  epoch_loss_train / args.trainset_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_size', type=int, help="'height or width, eg: 128'")
    parser.add_argument('image_depth', type=int, help="'depth of img, eg 16'")
    parser.add_argument('batch_size', type=int, default=100)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset_path', type=str, default='/nfs/managed_datasets/LIDC-IDRI/npy/average/')
    parser.add_argument('--scratch_path', type=str, default='/scratch/joelr/')
    parser.add_argument('--data_format', type=str, default='NCDHW')
    parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--image_channels', default=1, type=int)
    parser.add_argument('--final_output_channels', default=1, type=int)
    parser.add_argument('--first_output_channels', default=64, type=int)
    parser.add_argument('--testset_length', default=50, type=int)
    parser.add_argument('--trainset_length', default=1000, type=int)

    args = parser.parse_args()

    main(args)

