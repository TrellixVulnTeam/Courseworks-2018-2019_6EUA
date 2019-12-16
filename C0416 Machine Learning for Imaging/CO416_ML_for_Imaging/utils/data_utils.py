# Copyright 2019, Imperial College London
# 
# Tutorial for CO416 - Machine Learning for Imaging
#
# This file: Functions to manage data.

import numpy as np
import torchvision.datasets as dset

def get_mnist(data_dir, train, download=True):
    # data_dir: path to local directory where data is, or should be stored.
    # train: if True, return training data. If False, return test data.
    # download: if data not in data_dir, download it.
    data_set = dset.MNIST(root=data_dir, train=train, transform=None, download=True)
    # Students should only deal with numpy arrays, so that it's easy to follow.
    if train:
        data_x = np.asarray(data_set.train_data, dtype='uint8')
        data_y = np.asarray(data_set.train_labels, dtype='int16') # int64 by default
    else:
        data_x = np.asarray(data_set.test_data, dtype='uint8')
        data_y = np.asarray(data_set.test_labels, dtype='int16') # int64 by default
    return data_x, data_y


def normalize_int_whole_database(data):
    # data: shape [num_samples, H, W, C]
    mu = np.mean(data, axis=(0,1,2), keepdims=True) # Mean int of channel C, over samples and pixels.
    std = np.std(data, axis=(0,1,2), keepdims=True) # Returned shape: [1, 1, 1, C]
    norm_data = (data - mu) / std
    return norm_data
    