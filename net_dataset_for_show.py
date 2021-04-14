import math

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import manual_seed
from torch.utils.data import Dataset, random_split


class Net_dataset_show(Dataset):

    def __init__(self, data):
        self.data_close = data['Close']
        self.number_training = 150

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x_train = [0.0]
        y_train = [0.0]
        z_train = [0.0]
        for j in range(1, self.number_training - 1):
            percent = limit_one((self.data_close[j] / self.data_close[j - 1] - 1) * 100)
            if percent >= 0.0:
                x_train.append(percent)
                y_train.append(0.0)
            else:
                x_train.append(0.0)
                y_train.append(abs(percent))
            if j == 0:
                z_train.append(0.0)
            else:
                z_train.append(calculate_tendency(percent, x_train[-2]))

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        z_train = np.array(z_train)
        return x_train, y_train, z_train


def limit_one(number):
    if number > 1.0:
        return 1.0
    elif number < -1.0:
        return -1.0
    else:
        return number


def get_train_test_data(dataset, percent_train, number):
    train_size = math.ceil(len(dataset) * percent_train)
    test_size = len(dataset) - train_size
    data = random_split(dataset, [train_size, test_size], generator=manual_seed(number))
    return data


def calculate_tendency(new_percent, old_percent):
    if new_percent >= 0.0 and old_percent <= 0.0:
        return 0.0
    elif new_percent <= 0.0 and old_percent >= 0.0:
        return 0.0
    elif 0.0 <= old_percent <= new_percent:
        return 1.0
    elif 0.0 >= old_percent >= new_percent:
        return 1.0
    else:
        return abs(new_percent) / abs(old_percent)
