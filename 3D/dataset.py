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

        # test_npy_array = np.load(self.npy_files[0])[np.newaxis, ...]
        # self.shape = test_npy_array.shape
        # self.dtype = test_npy_array.dtype
        # del test_npy_array

    def __iter__(self):
        for path in self.npy_files:
            yield path

    def __getitem__(self, idx):
        return self.npy_files[idx]

    def __len__(self):
        return len(self.npy_files)




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
    dataset = dataset.make_one_shot_iterator()

    real_image_input = dataset.get_next()

    with tf.Session() as sess:
        sess.run(real_image_input)
