import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


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
        count = 0
        for j in range(idx, idx + size):
            if j == 0:
                percent = 0.0
            else:
                percent = limit_one((self.data_close[j] / self.data_close[j - 1] - 1) * 100)
            if len(x_train) != self.number_training:
                if percent > 0.0:
                    x_train.append(percent)
                    y_train.append(0.0)
                else:
                    x_train.append(0.0)
                    y_train.append(abs(percent))

                if count == 0:
                    var_percent = abs(percent) / 2
                    tendency = calculate_tendency(percent, 0.0)
                else:
                    var_percent = abs(percent - x_train[count - 1]) / 2
                    tendency = calculate_tendency(percent, x_train[count - 1])

                z_train.append(tendency)

            else:
                if percent >= 0.08:
                    sal_train.append(1)
                elif percent <= -0.08:
                    sal_train.append(2)
                else:
                    sal_train.append(0)
            count += 1

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        z_train = np.array(z_train)
        sal_train = np.array(sal_train)
        return x_train, y_train, z_train, sal_train


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


def limit_one(number):
    if number > 1.0:
        return 1.0
    elif number < -1.0:
        return -1.0
    else:
        return number

def normalize_data(list_number):
    d1 = 0.0
    d2 = 1.0
    x_min = -1
    x_max = 1
    list_normal = []
    for number in list_number:
        number = number
        list_normal.append(((((number - x_min) * (d2 - d1)) / (x_max - x_min)) + d1))
    return list_normal
