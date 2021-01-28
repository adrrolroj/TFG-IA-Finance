import datetime
import pandas as pd
import numpy as np

import torch


def train_model(model, dataloader, optimizer, criterion, epoch, cuda):
    torch.set_printoptions(precision=15)
    model.train()
    step = 0
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
            optimizer.zero_grad()
            output = model(x_train, y_train, z_train)
            loss = criterion(output, sal_train)
            loss.backward()
            optimizer.step()
            if step % 10 == 0 and i == 0:
                print(f'Epoca:{i}, Step:{step}, Perdida: {loss.sum()}')

    print("Entrenamiento finalizado")
    return model


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
        print(f'Ejecucion:{number}, Prediccion:{exit_values[int(output_exit)]}, Resultado real:{exit_values[real_exit]}, Precision acumulada:{success/(number+1)}')
        number += 1
        if number >= n_times:
            break
    precision = success / n_times
    print(f'Ejecucion finalizada, precision: {precision}')

