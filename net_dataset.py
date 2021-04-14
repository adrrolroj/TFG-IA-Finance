import math

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import manual_seed
from torch.utils.data import Dataset, random_split

class Net_dataset(Dataset):

    def __init__(self, file, size_sequence):
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df.index, utc=True)
        self.data_close = df['Close']
        self.number_training = size_sequence

    def __len__(self):
        return len(self.data_close) - (self.number_training + 1)

    def __getitem__(self, idx):
        torch.set_printoptions(precision=15)
        size = self.number_training + 1
        x_train = []
        y_train = []
        z_train = []
        sal_train = []
        for j in range(idx, idx + size):
            if len(x_train) != self.number_training:
                percent = limit_one((self.data_close[j] / self.data_close[j - 1] - 1) * 100)
                if percent >= 0.0:
                    x_train.append(percent)
                    y_train.append(0.0)
                else:
                    x_train.append(0.0)
                    y_train.append(abs(percent))
                if j == idx:
                    z_train.append(0.0)
                else:
                    z_train.append(calculate_tendency(percent, x_train[-2]))
            else:
                percent = limit_one((self.data_close[j] / self.data_close[j - 1] - 1) * 100)
                if percent >= 0.0677:
                    sal_train.append(1)
                elif percent <= -0.0601:
                    sal_train.append(2)
                else:
                    sal_train.append(0)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        z_train = np.array(z_train)
        sal_train = np.array(sal_train)
        return x_train, y_train, z_train, sal_train


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
