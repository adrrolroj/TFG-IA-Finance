
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class Net_dataset(Dataset):
    """Dataset con los datos de entrenamiento"""

    def __init__(self, file, size_sequence, size_exit):
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df.index, utc=True)
        self.data = df
        self.data_close = df['Close']
        self.number_training = size_sequence
        self.size_exit = size_exit

    def __len__(self):
        return len(self.data_close) - (self.number_training + self.size_exit)

    def __getitem__(self, idx):
        torch.set_printoptions(precision=15)
        size = self.number_training + self.size_exit
        x_train = []
        y_train = []
        z_train = []
        sal_train = []
        count = 0
        for j in range(idx, idx + size):
            count += 1
            if j == 0:
                percent = 0.0
            else:
                percent = (self.data_close[j] / self.data_close[j - 1] - 1) * 20
            if len(x_train) != self.number_training:
                x_train.append(limit_one(percent))
            else:
                sal_train.append(limit_one(percent))

            if count % 2 == 0:
                y_train.append(limit_one(x_train[count - 1] + x_train[count - 2]))
            if count % 4 == 0:
                z_train.append(limit_one(y_train[int(count / 2 - 1)] + y_train[int(count / 2 - 2)]))

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
