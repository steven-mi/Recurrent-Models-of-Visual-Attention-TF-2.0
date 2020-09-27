import numpy as np
import matplotlib.pyplot as plt

import glob
import os

from tqdm import tqdm

def load_bach_images(data_directory, resize=None):
    """
    This method returns the bach dataset as numpy array. Note that the test labels are not publicly available.

    Parameters
    ----------
    data_directory: String
        the path of the folder which contains the ICIAR2018_BACH_Challenge and ICIAR2018_BACH_Challenge_TestDataset folder

    Returns
    ----------
    (X_train, y_train), X_test, labels: (np.array, np.array), np.array, list
        The test and train dataset without test labels. In addition the names of the labels will be returned as list

    """
    # initializing folder, lists, ...
    image_directory = os.path.join(data_directory, 'ICIAR2018_BACH_Challenge/Photos/')
    labels = ['Normal', 'Benign', 'InSitu', 'Invasive']
    X_train, y_train, X_test = [], [], []
    # starting generating bach dataset
    for i, label in enumerate(labels):
        for image_path in tqdm(glob.glob(image_directory + label + '/*.tif')):
            # get the crops out of the image
            image =  plt.imread(image_path)
            if resize:
                import cv2
                image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
            # add the cropped images to the items of the file
            X_train.append(image)
            y_train.append(i)
    
    image_directory = os.path.join(data_directory, 'ICIAR2018_BACH_Challenge_TestDataset/Photos/')
    for image_path in tqdm(glob.glob(image_directory + '/*.tif')):
        image =  plt.imread(image_path)
        if resize:
            import cv2
            image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
        X_test.append(image)
    # write it into the file
    return (np.array(X_train), np.array(y_train)), np.array(X_test), labels

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
