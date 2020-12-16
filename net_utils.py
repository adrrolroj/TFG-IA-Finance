import datetime
from random import randint
import pandas as pd
import mplfinance as mpf

import torch


def train_model(model, dataloader, optimizer, criterion, epoch):
    torch.set_printoptions(precision=15)
    step = 0
    model.train()
    w = 0
    for x_train, y_train, z_train, sal_train in dataloader:
        w += 1
        if w > 2000:
            break
        step += 1
        x_train = x_train.view(x_train.shape[1]).float()
        y_train = y_train.view(y_train.shape[1]).float()
        z_train = z_train.view(z_train.shape[1]).float()
        sal_train = sal_train.view(1, sal_train.shape[0]).float()
        for i in range(epoch):
            optimizer.zero_grad()
            output = model(x_train, y_train, z_train)
            loss = criterion(output, sal_train)
            loss.backward()
            optimizer.step()
            print(f'Epoca:{i}, Step:{step}, Perdida: {loss.sum()}')
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

    fig = mpf.figure(style='charles', figsize=(12, 8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 3)

    ap = [mpf.make_addplot(df_y, type='candle', ax=ax2, ylabel='Salida real')]
    mpf.plot(df_out, ax=ax1, addplot=ap, type='candle', ylabel='Salida de la red')
    mpf.show()

def evaluate_model(model, dataloader, n_times):
    torch.set_printoptions(precision=20)
    acumulated_error = 0.0
    number = 0
    for x_train, y_train, z_train, sal_train in dataloader:
        x_train = x_train.view(x_train.shape[1]).float()
        y_train = y_train.view(y_train.shape[1]).float()
        z_train = z_train.view(z_train.shape[1]).float()
        sal_train = sal_train.numpy()
        output = model(x_train, y_train, z_train).detach().numpy()
        error = 0.0
        for i in range(len(sal_train)):
            error = abs(output[i]/20 - sal_train[i]/20)
            acumulated_error += error
        print(f'Ejecucion:{number}, Prediccion:{output[0]/20}, Resultado real:{sal_train[0]/20}, Error:{error}')
        number += 1
        if number >= n_times:
            break
    media = acumulated_error / n_times
    print(f'Ejecucion finalizada, Media del error: {media}')


def dataframe_constructor(number, percent):
    percent = percent / 20
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
