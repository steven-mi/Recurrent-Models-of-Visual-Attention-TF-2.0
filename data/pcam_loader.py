import os
from tensorflow.keras.utils import HDF5Matrix
import matplotlib.pyplot as plt
import numpy as np

def load_pcam_images(path_to_h5_files):
    """
    This method returns the bach dataset as numpy array. Note that the test labels are not publicly available.

    Parameters
    ----------
    data_directory: String
        the path of the folder which contains the .h5 files which can be downloaded from Github
        https://github.com/basveeling/pcam

    Returns
    ----------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) labels: (np.array, np.array), (np.array, np.array), (np.array, np.array)
        The test and train dataset without test labels. In addition the names of the labels will be returned as list

    """
    X_train_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_train_x.h5')
    y_train_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_train_y.h5')
    X_val_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_valid_x.h5')
    y_val_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_valid_y.h5')
    X_test_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_test_x.h5')
    y_test_path = os.path.join(path_to_h5_files, 'camelyonpatch_level_2_split_test_y.h5')
    
    X_train = HDF5Matrix(X_train_path, 'x').data[()]
    y_train = HDF5Matrix(y_train_path, 'y').data[()]
    
    X_val = HDF5Matrix(X_val_path, 'x').data[()]
    y_val = HDF5Matrix(y_val_path, 'y').data[()]

    X_test = HDF5Matrix(X_test_path, 'x').data[()]
    y_test = HDF5Matrix(y_test_path, 'y').data[()]
    
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def minibatcher(inputs, targets, batchsize, shuffle=False):
    """
    This method creates a iterable batcher
    FROM: https://github.com/deworrall92/harmonicConvolutions/blob/master/MNIST-rot/run_mnist.py

    Parameters
    ----------
    inputs, targets: np.arrays
        The input (e.g. training images) and targets (e.g. training labels) which the minibatcher should make batches of
    batchsize: int
        The size of a single batch
    shuffle: boolean (default False)
        If its true then the batches will be random

    Returns
    ----------
    minibatcher: iterable
        In order to use it you'll need to iterate though the minibatcher e.g.:

        batcher = minibatcher(X, y, 200, True)
        for X_batch, y_batch in batcher:
            print(X_batch.shape) -> (batchsize, ...)
            print(y_batch.shape) -> (batchsize, ...)
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

