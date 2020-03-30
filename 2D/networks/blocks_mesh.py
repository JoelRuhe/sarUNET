from networks.ops_mesh import *
import mesh_tensorflow as mtf

kernel = (3, 3)

def contracting_block(x, output_shape, filtername):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        out1_dim = mtf.Dimension(filtername, output_shape)
        x = conv2d(x, out1_dim, kernel)

    with tf.variable_scope("conv_2"):
        out2_dim = mtf.Dimension(filtername+"1", output_shape)
        x = conv2d(x, out2_dim, kernel)

    return x


def bottleneck(x, output_shape, filtername):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        out_dim1 = mtf.Dimension(filtername+"1", output_shape)
        x = conv2d(x, out_dim1, kernel)

    with tf.variable_scope("conv_2"):
        out_dim2 = mtf.Dimension(filtername+"2", output_shape)
        x = conv2d(x, out_dim2, kernel)

    with tf.variable_scope("conv_trans"):
        print(output_shape)
        out_dim3 = mtf.Dimension(filtername+"3", (output_shape//2))
        print(out_dim3, 'OUT_DIM3 L;ASLASKDFJL;AKFS')
        x = conv2d_transpose(x, out_dim3, kernel)

    print(x.shape, 'X.SHAPE')

    return x


def expansion_block(x, mid_channels, out_channels, filtername):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        out_dim1 = mtf.Dimension(filtername, mid_channels)
        x = conv2d(x, out_dim1, kernel)

    with tf.variable_scope("conv_2"):
        out_dim2 = mtf.Dimension(filtername+"1", mid_channels)
        x = conv2d(x, out_dim2, kernel)

    with tf.variable_scope("conv_trans"):
        out_dim3 = mtf.Dimension(filtername+"2", out_channels)
        x = conv2d_transpose(x, out_dim3, kernel)

    return x


def final_layer(x, mid_channels, out_channels, filtername):
    """TODO: Implement with BatchNorm2d"""

    with tf.variable_scope("conv_1"):
        out_dim1 = mtf.Dimension(filtername, mid_channels)
        x = conv2d(x, out_dim1, kernel)

    with tf.variable_scope("conv_2"):
        out_dim2 = mtf.Dimension(filtername+"1", mid_channels)
        x = conv2d(x, out_dim2, kernel)

    with tf.variable_scope("conv_3"):
        out_dim3 = mtf.Dimension("channel", out_channels)
        x = conv2d(x, out_dim3, kernel)

    return x


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        lst = list()
        arr = np.empty(4)
        list_string = list()

        # bypass = tf.cast(bypass, tf.Tensor)
        # print(type(bypass))

        print(type(upsampled))
        print(type(bypass))
        print(bypass.shape)
        print(upsampled.shape)
        upsampled_w = str((upsampled.shape[1]))

        upsampled_h = str((upsampled.shape[2]))
        bypass_w = str(bypass.shape[1])
        bypass_h = str(bypass.shape[2])

        lst.append(upsampled_w)
        lst.append(upsampled_h)
        lst.append(bypass_w)
        lst.append(bypass_h)

        for i in range(4):
            print(i)
            split1 = lst[i].split("size")
            print(split1[1])
            split2 = split1[1].split("=")
            print(split2[1])
            split3 = split2[1].split(")")
            print(split3)
            print(split3[0])
            print(str(split3[0]))
            split3 = str(split3[0])
            arr[i] = split3

        print(arr[0])

        upsampled_w = int(arr[0])
        upsampled_h = int(arr[1])
        bypass_w = int(arr[2])
        bypass_h = int(arr[3])

        w = (bypass_w - upsampled_w) // 2
        h = (bypass_h - upsampled_h) // 2

        print(w)
        print(h)
        paddings = ([-h, -h], [-w, -w])
        print(len(paddings))
        # bypass = mtf.pad(bypass, ([0, 0], [-h, -h], [-w, -w], [0, 0]), "CONSTANT")
        bypass = mtf.pad(bypass, paddings=paddings, dim_name="col_blocks")

    return mtf.concat([upsampled, bypass], 3)
