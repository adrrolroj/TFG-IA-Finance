import random

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class False_dataset(Dataset):

    def __init__(self, size_sequence, number):
        self.size_sequence = size_sequence
        self.number = number

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        rd = random.randint(0, 2)
        sal_train = []
        if rd == 0:
            x_train = [0] * self.size_sequence
            y_train = [0] * self.size_sequence
            sal_train.append(0)
        elif rd == 1:
            x_train = [1] * self.size_sequence
            y_train = [0] * self.size_sequence
            sal_train.append(1)
        else:
            x_train = [0] * self.size_sequence
            y_train = [1] * self.size_sequence
            sal_train.append(2)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        sal_train = np.array(sal_train)
        return x_train, y_train, sal_train
