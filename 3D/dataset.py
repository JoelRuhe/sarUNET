import glob
import numpy as np
import os
import shutil
import time
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tqdm import tqdm
from os import listdir
from os.path import isfile, join



class NumpyDataset:

    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        if copy_files and is_correct_phase:
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        # del test_npy_array

    def __iter__(self):
        for path in self.npy_files:
            yield np.load(path)[np.newaxis, ...]

    def __getitem__(self, idx):
        return np.load(self.npy_files[idx])[np.newaxis, ...]

    def __len__(self):
        return len(self.npy_files)

class NumpyPathDataset:
    def __init__(self, npy_dir, scratch_dir, copy_files, is_correct_phase):
        super(NumpyPathDataset, self).__init__()
        self.npy_files = glob.glob(npy_dir + '*.npy')
        print(f"Length of dataset: {len(self.npy_files)}")

        if scratch_dir is not None:
            if scratch_dir[-1] == '/':
                scratch_dir = scratch_dir[:-1]

        self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
        if copy_files and is_correct_phase:
            os.makedirs(self.scratch_dir, exist_ok=True)
            print("Copying files to scratch...")
            for f in self.npy_files:
                # os.path.isdir(self.scratch_dir)
                if not os.path.isfile(os.path.normpath(scratch_dir + f)):
                    shutil.copy(f, os.path.normpath(scratch_dir + f))

        while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_files):
            time.sleep(1)

        self.scratch_files = glob.glob(self.scratch_dir + '/*.npy')
        assert len(self.scratch_files) == len(self.npy_files)

        test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        self.shape = test_npy_array.shape
        self.dtype = test_npy_array.dtype
        del test_npy_array

    def __iter__(self):
        for path in self.npy_files:
            yield path

    def __getitem__(self, idx):
        return self.npy_files[idx]

    def __len__(self):
        return len(self.npy_files)

# class NumpyPathDataset:
#
#     def __init__(self, npy_dir, scratch_dir, train, train_size, copy_files: bool, is_correct_phase):
#         super(NumpyPathDataset, self).__init__()
#         self.npy_files = glob.glob(npy_dir + '*.npy')
#
#         # npy_train_length and npy_test_length are calculated based on the train_size percentage flag, default: 90%
#         # train size, 10% test size
#         npy_length = int(len(self.npy_files))
#         npy_train_length = int(npy_length * train_size)
#         npy_test_length = int(npy_length * (1-train_size))
#
#         self.npy_train_files = self.npy_files[0: npy_train_length]
#         self.npy_test_files = self.npy_files[npy_train_length: (npy_train_length + npy_test_length) + 1]
#         self.is_train = train
#
#         print("Lenght of training files = " + str(len(self.npy_train_files)))
#         print("Lenght of test files = " + str(len(self.npy_test_files)))
#
#         if scratch_dir is not None:
#             if scratch_dir[-1] == '/':
#                 scratch_dir = scratch_dir[:-1]
#
#         self.scratch_dir = os.path.normpath(scratch_dir + npy_dir) if is_correct_phase else npy_dir
#
#         if train:
#             os.makedirs(self.scratch_dir, exist_ok=True)
#             if copy_files and is_correct_phase:
#                 print("Copying training files to scratch...")
#                 for f in tqdm(self.npy_train_files):
#                     if not os.path.isfile(os.path.normpath(scratch_dir + f)):
#                         shutil.copy(f, os.path.normpath(scratch_dir + f))
#
#                 while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_train_files):
#                     print(len(glob.glob(self.scratch_dir + '/*.npy')))
#                     print(len(self.npy_train_files))
#                     time.sleep(1)
#
#                 self.scratch_files_train = glob.glob(self.scratch_dir + '/*.npy')
#                 print(len(self.scratch_files_train))
#                 print(len(self.npy_train_files))
#                 assert len(self.scratch_files_train) == len(self.npy_train_files)
#
#             if not train:
#                 print("Copying test files to scratch...")
#                 for f in tqdm(self.npy_test_files):
#                     if not os.path.isfile(os.path.normpath(scratch_dir + f)):
#                         shutil.copy(f, os.path.normpath(scratch_dir + f))
#
#                 while len(glob.glob(self.scratch_dir + '/*.npy')) < len(self.npy_test_files):
#                     time.sleep(1)
#
#                 self.scratch_files_test = glob.glob(self.scratch_dir + '/*.npy')
#                 print(len(self.scratch_files_test))
#                 print(len(self.npy_test_files))
#                 assert len(self.scratch_files_test) == len(self.npy_test_files)
#
#
#     def __len__(self):
#         if self.is_train:
#             return len(self.npy_train_files)
#         else:
#             return len(self.npy_test_files)
#
#     def get_data(self):
#         if self.is_train:
#             return self.npy_train_files
#         else:
#             return self.npy_test_files



def npy_data(npy_dir, scratch_dir, train_size, train, copy_files, is_correct_phase):
    npy_data = NumpyPathDataset(npy_dir, scratch_dir, train, train_size, copy_files, is_correct_phase)

    dataset = tf.data.Dataset.from_tensor_slices(npy_data.get_data())

    dataset = dataset.shuffle(len(npy_data))

    return dataset, npy_data


if __name__ == '__main__':

    npy_data = NumpyPathDataset('/lustre4/2/managed_datasets/LIDC-IDRI/npy/average/4x4/', '/scratch-local', copy_files=True,
                                is_correct_phase=True)

    dataset = tf.data.Dataset.from_tensor_slices(npy_data.get_data())


    # Lay out the graph.
    dataset = dataset.shuffle(len(npy_data))
    # dataset = dataset.map(lambda x: tf.py_function(func=load, inp=[x], Tout=tf.uint16), num_parallel_calls=AUTOTUNE)
    # dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / 1024 - 1, num_parallel_calls=AUTOTUNE)
    # dataset = dataset.batch(256, drop_remainder=True)
    # dataset = dataset.prefetch(AUTOTUNE)
    # dataset = dataset.repeat()
    dataset = dataset.make_one_shot_iterator()

    real_image_input = dataset.get_next()

    with tf.Session() as sess:
        sess.run(real_image_input)
