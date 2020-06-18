import numpy as np
import tensorflow as tf
from networks.blocks import *
from networks.ops import *


def unet(x, args):
    tensor_list = list()

    # determine how many convolutions has to be done based on image size
    num_downsamples = int(np.log2(args.image_size) - np.log2(args.lowest_size))

    # CONTRACTION

    for i in range(num_downsamples):
        with tf.variable_scope("contract" + str(i)):
            x = contracting_block(x, args.first_output_channels, args.data_format, activation=args.activation,
                                  param=args.leakiness)
            # add tensor to list for later usage in Expanse
            tensor_list.append(x)
            x = maxpool3d(x, (2, 2, 2), (2, 2, 2), args.data_format, padding="SAME")
            if args.data_format == "NCDHW":
                args.first_output_channels = x.shape[1].value * 2
            else:
                args.first_output_channels = x.shape[4].value * 2

    # BOTTLENECK
    if args.data_format == "NCDHW":
        x = bottleneck(x, x.shape[1].value * 2, args.data_format, activation=args.activation, param=args.leakiness)
    else:
        x = bottleneck(x, x.shape[4].value * 2, args.data_format, activation=args.activation, param=args.leakiness)

    # EXPANSE
    print(x.shape, 'decode first')
    for i in reversed(range(len(tensor_list) - 1)):
        x = crop_and_concat(x, tensor_list[i + 1], args.data_format, crop=True)
        print(x.shape, 'concat')
        with tf.variable_scope("expanse" + str(i)):
            if args.data_format == "NCDHW":
                print("TEST2")
                x = expansion_block(x, x.shape[1].value // 2, x.shape[1].value // 4, args.data_format,
                                    activation=args.activation, param=args.leakiness)
                print(x, 'X.SHAPE 2')
            else:
                x = expansion_block(x, x.shape[4].value // 2, x.shape[4].value // 4, args.data_format,
                                    activation=args.activation, param=args.leakiness)

            print(x.shape, 'decode')

    # FINAL

    x = crop_and_concat(x, tensor_list[0], args.data_format, crop=True)

    with tf.variable_scope("final"):
        if args.data_format == "NCDHW":
            x = final_layer(x, x.shape[1].value // 2, args.final_output_channels, args.data_format,
                            activation=args.activation, param=args.leakiness)
        else:
            x = final_layer(x, x.shape[4].value // 2, args.final_output_channels, args.data_format,
                            activation=args.activation, param=args.leakiness)
    return x