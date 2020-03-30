# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST using Mesh TensorFlow and TF Estimator.
This is an illustration, not a good model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
# import mnist_dataset as dataset  # local file import
import tensorflow.compat.v1 as tf
import argparse
import numpy as np
from tensorflow.data.experimental import AUTOTUNE
from dataset import imagenet_dataset
from networks.blocks_mesh import *
from tensorflow.python.platform import flags
import time

import os


FLAGS = flags.FLAGS


# tf.flags.DEFINE_string("data_dir", "data/",
#                        "Path to directory containing the MNIST dataset")
# tf.flags.DEFINE_string("model_dir", "/tmp/mnist_model", "Estimator model_dir")
# tf.flags.DEFINE_integer("batch_size", 200,
#                         "Mini-batch size for the training. Note that this "
#                         "is the global batch size and not the per-shard batch.")
# tf.flags.DEFINE_integer("hidden_size", 512, "Size of each hidden layer.")
# tf.flags.DEFINE_integer("train_epochs", 40, "Total number of training epochs.")
# tf.flags.DEFINE_integer("epochs_between_evals", 1,
#                         "# of epochs between evaluations.")
# tf.flags.DEFINE_integer("eval_steps", 0,
#                         "Total number of evaluation steps. If `0`, evaluation "
#                         "after training is skipped.")
# tf.flags.DEFINE_string("mesh_shape", "b1:2;b2:2", "mesh shape")
# tf.flags.DEFINE_string("layout", "row_blocks:b1;col_blocks:b2",
#                        "layout rules")

FLAGS = tf.flags.FLAGS

def mnist_model(image, labels, mesh):
  """The model.
  Args:
    image: tf.Tensor with shape [batch, 28*28]
    labels: a tf.Tensor with shape [batch] and dtype tf.int32
    mesh: a mtf.Mesh
  Returns:
    logits: a mtf.Tensor with shape [batch, 10]
    loss: a mtf.Tensor with shape []
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  row_blocks_dim = mtf.Dimension("row_blocks", 4)
  col_blocks_dim = mtf.Dimension("col_blocks", 4)
  rows_dim = mtf.Dimension("rows_size", 7)
  cols_dim = mtf.Dimension("cols_size", 7)

  classes_dim = mtf.Dimension("classes", 10)
  one_channel_dim = mtf.Dimension("one_channel", 1)

  x = mtf.import_tf_tensor(
      mesh, tf.reshape(image, [FLAGS.batch_size, 4, 7, 4, 7, 1]),
      mtf.Shape(
          [batch_dim, row_blocks_dim, rows_dim,
           col_blocks_dim, cols_dim, one_channel_dim]))
  x = mtf.transpose(x, [
      batch_dim, row_blocks_dim, col_blocks_dim,
      rows_dim, cols_dim, one_channel_dim])

  # add some convolutional layers to demonstrate that convolution works.
  filters1_dim = mtf.Dimension("filters1", 16)
  filters2_dim = mtf.Dimension("filters2", 16)
  f1 = mtf.relu(mtf.layers.conv2d_with_blocks(
      x, filters1_dim, filter_size=[9, 9], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv0"))
  f2 = mtf.relu(mtf.layers.conv2d_with_blocks(
      f1, filters2_dim, filter_size=[9, 9], strides=[1, 1], padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim, name="conv1"))
  x = mtf.reduce_mean(f2, reduced_dim=filters2_dim)

  # add some fully-connected dense layers.
  hidden_dim1 = mtf.Dimension("hidden1", FLAGS.hidden_size)
  hidden_dim2 = mtf.Dimension("hidden2", FLAGS.hidden_size)

  h1 = mtf.layers.dense(
      x, hidden_dim1,
      reduced_dims=x.shape.dims[-4:],
      activation=mtf.relu, name="hidden1")
  h2 = mtf.layers.dense(
      h1, hidden_dim2,
      activation=mtf.relu, name="hidden2")
  logits = mtf.layers.dense(h2, classes_dim, name="logits")
  if labels is None:
    loss = None
  else:
    labels = mtf.import_tf_tensor(
        mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, mtf.one_hot(labels, classes_dim), classes_dim)
    loss = mtf.reduce_mean(loss)
  return logits, loss


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  tf.logging.info("features = %s labels = %s mode = %s params=%s" %
                  (features, labels, mode, params))
  global_step = tf.train.get_global_step()
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  logits, loss = mnist_model(features, labels, mesh)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  mesh_size = mesh_shape.size
  mesh_devices = [""] * mesh_size
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  restore_hook = mtf.MtfRestoreHook(lowering)

  tf_logits = lowering.export_to_tf_tensor(logits)
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf.summary.scalar("loss", tf_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)
    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=2,
        defer_build=False, save_relative_paths=True)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    saver_listener = mtf.MtfCheckpointSaverListener(lowering)
    saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.model_dir,
        save_steps=1000,
        saver=saver,
        listeners=[saver_listener])

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(tf_logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(tf_loss, "cross_entropy")
    tf.identity(accuracy[1], name="train_accuracy")

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar("train_accuracy", accuracy[1])

    # restore_hook must come before saver_hook
    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
        training_chief_hooks=[restore_hook, saver_hook])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(tf_logits, axis=1),
        "probabilities": tf.nn.softmax(tf_logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[restore_hook],
        export_outputs={
            "classify": tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf_loss,
        evaluation_hooks=[restore_hook],
        eval_metric_ops={
            "accuracy":
            tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
        })


def run_mnist():
  """Run MNIST training and eval loop."""
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=FLAGS.model_dir)

  # Set up training and evaluation input functions.
  def train_input_fn():
    """Prepare data for training."""

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    # ds = dataset.train(FLAGS.data_dir)
    # ds_batched = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size)
    #
    # # Iterate through the dataset a set number (`epochs_between_evals`) of times
    # # during each training session.
    # ds = ds_batched.repeat(FLAGS.epochs_between_evals)
    ds = None
    return ds

  def eval_input_fn():
      print('nothing')
    # return dataset.test(FLAGS.data_dir).batch(
    #     FLAGS.batch_size).make_one_shot_iterator().get_next()

  # Train and evaluate model.
  for _ in range(FLAGS.train_epochs // FLAGS.epochs_between_evals):
    mnist_classifier.train(input_fn=train_input_fn, hooks=None)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("\nEvaluation results:\n\t%s\n" % eval_results)


def forward(x, args, row_blocks_dim, col_blocks_dim, lowering):
    counter = 0
    tensor_list = list()
    output_shape = args.first_output_channels

    # determine how many convolutions should be done based on image size.
    while args.image_size > 4:
        args.image_size = args.image_size // 2
        counter = counter + 1

    # ------------------ CONTRACTION ----------------

    for i in range(counter):
        with tf.variable_scope("contract" + str(i)):
            print(x.shape, "conv:" + str(i))
            output_shape = args.base_channels if i == 0 else output_shape * 2
            x = contracting_block(x, output_shape, "out_channels")
            tensor_list.append(x)
            x = mtf.layers.max_pool2d(x, ksize=(2, 2))

    # ------------------ BOTTLENECK ----------------

    x = bottleneck(x, output_shape*2, "filtername")

    # ------------------ EXPANSE ----------------
    print(len(tensor_list))
    print('EXPANSE')
    for i in reversed(range(len(tensor_list) - 1)):
        print(x.shape, "tranpose: " + str(i))
        # x = crop_and_concat(x, tensor_list[i + 1], crop=True)
        if i is (len(tensor_list)-2):
            print(output_shape//2, 'OUTPUT SHAPE //2')
            with tf.variable_scope("expanse" + str(i)):
                x = expansion_block(x, output_shape // 2, output_shape // 2, "out_channels")
        else:
            with tf.variable_scope("expanse" + str(i)):
                x = expansion_block(x, output_shape//2, output_shape//4, "out_channels")

            output_shape = output_shape//2

    print(x.shape)

    # ------------------ FINAL ----------------

    # x = crop_and_concat(x, tensor_list[0], crop=True)

    print('FINAL')
    with tf.variable_scope("final"):
        x = final_layer(x, output_shape // 2, args.final_output_channels, "out_channels")

    print(x.shape)

    return x


def main(args):

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    mesh_shape = mtf.convert_to_shape(args.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(args.layout)
    mesh_size = mesh_shape.size
    mesh_devices = [""] * mesh_size
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, mesh_devices)

    print(mesh_impl.shape, '2')
    print(mesh_devices, 'mesh_devices')

    batch_dim = mtf.Dimension("batch", args.batch_size)
    row_blocks_dim = mtf.Dimension("row_blocks", args.image_size)
    col_blocks_dim = mtf.Dimension("col_blocks", args.image_size)
    # rows_dim = mtf.Dimension("rows_size", 8)
    # cols_dim = mtf.Dimension("cols_size", 8)
    channel_dim = mtf.Dimension("channel", args.RGB)
    out_channels_dim = mtf.Dimension("out_channels_dim", 3)

    # ------------------ DATASET ----------------

    # Train data pipeline
    is_train = tf.placeholder(tf.bool, name="is_train")
    dataset_train, imagenet_data_train = imagenet_dataset(args.dataset_root, args.scratch_path, args.image_size,
                                                          train=True, copy_files=True, num_labels=args.num_labels)
    dataset_train = dataset_train.batch(args.batch_size, drop_remainder=True)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.prefetch(AUTOTUNE)
    dataset_train = dataset_train.make_one_shot_iterator()

    # Test data pipeline
    dataset_test, imagenet_data_test = imagenet_dataset(args.dataset_root, args.scratch_path, args.image_size,
                                                        train=False, copy_files=True, num_labels=args.num_labels)
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

    image_input = mtf.import_tf_tensor(
        mesh, tf.reshape(image_input, [args.batch_size, args.image_size, args.image_size, args.RGB]),
        mtf.Shape([batch_dim, row_blocks_dim, col_blocks_dim, channel_dim]))

    y = image_input
    print(type(y), 'TYPE Y')


    # labels = mtf.import_tf_tensor(
    #     mesh, tf.reshape(labels, [FLAGS.batch_size]), mtf.Shape([batch_dim]))
    # loss = mtf.layers.softmax_cross_entropy_with_logits(
    #     logits, mtf.one_hot(labels, classes_dim), classes_dim)
    # loss = mtf.reduce_mean(loss)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    prediction = forward(image_input, args, row_blocks_dim, col_blocks_dim, lowering)
    print(prediction.shape.dims)
    # ------------------ OPTIM -----------------

    loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(logits=prediction, targets=y, vocab_dim=channel_dim))
    # lr_scaler = hvd.size() if args.horovod else 1
    # optimizer = mtf.optimize.AdafactorOptimizer(learning_rate=args.learning_rate)
    # update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    # if args.horovod:
    #     optimizer = hvd.DistributedOptimizer(optimizer)

    # train_step = optimizer.minimize(loss)
    image_input = lowering.export_to_tf_tensor(image_input)
    y = lowering.export_to_tf_tensor(y)
    prediction = lowering.export_to_tf_tensor(prediction)
    loss = lowering.export_to_tf_tensor(loss)

    # ------------- SUMMARIES -------------

    # Transpose to valid image format.
    x_input = tf.transpose(image_input, perm=[0,2,3,1])
    y = tf.transpose(y, perm=[0,2,3,1])
    prediction = tf.transpose(prediction, perm=[0,2,3,1])

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

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        # if verbose:
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', timestamp)
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph, session=sess)

        for epoch in range(args.epochs):
            epoch_loss_train = 0
            epoch_loss_test = 0
            num_train_steps = len(imagenet_data_train) // args.batch_size
            num_test_steps = len(imagenet_data_test) // args.batch_size
            # print(global_batch_size, 'global batch size')
            # print(len(imagenet_data_test), 'LENGTH IMAGENET DATA TEST')
            # print(num_train_steps, 'NUM TRAIN STEPS')
            # print(num_test_steps, 'NUM TEST STEPS')
            train = True
            for i in range(num_train_steps):
                _, summary, c = sess.run([update_ops, image_summary_train, loss], feed_dict={is_train: train})

                # if i % args.logging_interval == 0 and verbose:
                global_step = (epoch * num_train_steps * args.batch_size) + i * args.batch_size
                writer.add_summary(summary, global_step)
                writer.flush()
                epoch_loss_train += c

            train = False
            for i in range(num_test_steps):
                c = sess.run(loss, feed_dict={is_train: train})
                # print(c, 'test loss c')
            # if verbose:
            epoch_loss_test += c
            # print(epoch_loss_test, 'epoch loss')
            writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag='loss_test', simple_value=epoch_loss_test / num_test_steps)]), global_step)
            test_image_summary = sess.run(image_summary_test, feed_dict={is_train: train})
            writer.add_summary(test_image_summary, global_step)
            writer.flush()

            # if verbose:
            print(f'Epoch [{epoch}/{args.epochs}]\t'
                  f'Train Loss: {epoch_loss_train / num_train_steps}\t'
                  f'Test Loss: {epoch_loss_test / num_test_steps}\t')


if __name__ == "__main__":
  # tf.disable_v2_behavior()
  # tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run()

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True, choices=['imagenet'])
  parser.add_argument('--dataset_root', type=str, required=True)
  parser.add_argument('--image_size', type=int, help="'height or width, eg: 128'", required=True)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--scratch_path', type=str, default=f'/scratch/{os.environ["USER"]}/')
  parser.add_argument('--loss_fn', default='mean_squared_error', choices=['mean_squared_error'])
  parser.add_argument('--noise_strength', type=float, default=0.03)
  parser.add_argument('--activation', type=str, default='leaky_relu')
  parser.add_argument('--leakiness', type=float, default=0.2)
  parser.add_argument('--num_labels', default=2, type=int)
  parser.add_argument('--epochs', default=100000, type=int)
  parser.add_argument('--base_channels', default=64, type=int, help='Controls network complexity (parameters).')
  parser.add_argument('--final_output_channels', default=3, type=int)
  parser.add_argument('--first_output_channels', default=64, type=int)
  parser.add_argument('--learning_rate', default=1e-5, type=float)
  parser.add_argument('--logging_interval', default=8, type=int)
  parser.add_argument('--RGB', default=3, type=int)
  parser.add_argument('--horovod', default=False, action='store_true')
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument('--gpu', action='store_true')
  parser.add_argument('--sampled_2d_slices',  type=bool, default=False, help='Whether to build model on 2D CT slices instead of 3D.')
  parser.add_argument('--mesh_shape', type=str, default="rows:4, columns:4")
  parser.add_argument('--layout',  type=str, default="row_blocks:4, col_blocks:4")

  # flags.DEFINE_boolean('sampled_2d_slices', False,
  #                      'Whether to build model on 2D CT slices instead of 3D.')

  args = parser.parse_args()
  main(args)