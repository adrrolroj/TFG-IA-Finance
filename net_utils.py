import datetime
from random import randint
import pandas as pd
import mplfinance as mpf
import numpy as np

import torch


def train_model(model, dataloader, optimizer, criterion, epoch, cuda):
    torch.set_printoptions(precision=15)
    step = 0
    model.train()
    number = 0
    success = 0
    for x_train, y_train, z_train, sal_train in dataloader:
        step += 1
        x_train = x_train.view(x_train.shape[1]).float()
        y_train = y_train.view(y_train.shape[1]).float()
        z_train = z_train.view(z_train.shape[1]).float()
        sal_train = sal_train.view(1)
        if cuda:
            x_train = x_train.to('cuda')
            y_train = y_train.to('cuda')
            z_train = z_train.to('cuda')
            sal_train = sal_train.to('cuda')
        for i in range(epoch):
            number += 1
            optimizer.zero_grad()
            output = model(x_train, y_train, z_train)
            loss = criterion(output, sal_train)
            loss.backward()
            optimizer.step()
            if cuda:
                output = output.cpu()
            if np.argmax(output.detach().numpy()) == sal_train.item():
                success += 1
            if step % 10 == 0 and i == 0:
                print(f'Epoca:{i}, Step:{step}, Perdida: {loss.sum()}, Precision: {success/number}')
                success = 0
                number = 0
    print("Entrenamiento finalizado")
    return model


def evaluate_model_graphic(model, dataloader):
    dataset = dataloader.dataset
    n_samples = len(dataset)
    random_index = randint(0, n_samples)
    x_test, y_test, z_test, sal_test = dataset[random_index]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    z_test = torch.from_numpy(z_test)
    sal_test = torch.from_numpy(sal_test)
    x_test = x_test.view(x_test.shape[0]).float()
    y_test = y_test.view(y_test.shape[0]).float()
    z_test = z_test.view(z_test.shape[0]).float()

    output = model(x_test, y_test, z_test)

    df_y = dataframe_constructor(100.0, sal_test.item())
    df_out = dataframe_constructor(100.0, output.item())
    df = pd.concat([df_y, df_out])
    fig = mpf.figure(style='charles', figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)

    mpf.plot(df, ax=ax1, type='candle', ylabel='(Izq) S.Real (Der) S.Red')
    mpf.show()


def evaluate_model(model, dataloader, n_times, cuda):
    torch.set_printoptions(precision=20)
    success = 0
    number = 0
    for x_train, y_train, z_train, sal_train in dataloader:
        x_train = x_train.view(x_train.shape[1]).float()
        y_train = y_train.view(y_train.shape[1]).float()
        z_train = z_train.view(z_train.shape[1]).float()
        if cuda:
            x_train = x_train.to('cuda')
            y_train = y_train.to('cuda')
            z_train = z_train.to('cuda')
            output = model(x_train, y_train, z_train)
            output = output.cpu().detach().numpy()
        else:
            output = model(x_train, y_train, z_train)
            output = output.detach().numpy()
        real_exit = sal_train.item()
        output_exit = np.argmax(output)
        if output_exit == real_exit:
            success += 1
        exit_values = ('CERO', 'SUBE', 'BAJA')
        print(f'Ejecucion:{number}, Prediccion:{exit_values[output_exit]}, Resultado real:{exit_values[real_exit]}, Precision acumulada:{success/(number+1)}')
        number += 1
        if number >= n_times:
            break
    precision = success / n_times
    print(f'Ejecucion finalizada, precision: {precision}')


def dataframe_constructor(number, percent):
    percent = convert_normal_data_to_percent(percent)
    data = {'Datetime': [datetime.datetime.now()],
            'Open': [number],
            'High': [number],
            'Low': [number],
            'Close': [number + number * percent],
            'Adj Close': [number + number * percent],
            'Volume': [0.0]}
    df = pd.DataFrame(data=data)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def convert_normal_data_to_percent(number):
    d1 = 0.0
    d2 = 1.0
    x_min = -1
    x_max = 1
    percent = ((number * (x_max - x_min) - d1 * (x_max - x_min)) / (d2 - d1)) + x_min
    percent = percent
    return percent
