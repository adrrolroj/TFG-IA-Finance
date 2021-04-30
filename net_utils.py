import math

import numpy as np

import torch
from random import randint
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from nets import AdaptativeNet
from stac_files.nonparametric_tests import friedman_test, holm_test


def train_model(model, trainloader, testloader, optimizer, criterion, epoch, cuda):
    if cuda:
        model.to('cuda')
    for epoch in range(epoch):
        model.train()
        for i, data in enumerate(trainloader, 0):
            x_train, y_train, z_train, sal_train = data
            x_train = x_train.float()
            y_train = y_train.float()
            z_train = z_train.float()
            if cuda:
                x_train = x_train.to('cuda')
                y_train = y_train.to('cuda')
                z_train = z_train.to('cuda')
                sal_train = sal_train.to('cuda')

            optimizer.zero_grad()
            output = model(x_train, y_train, z_train)
            sal_train = sal_train.view(sal_train.shape[0])
            loss = criterion(output, sal_train)
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                _, predicted = torch.max(output.data, 1)
                total = sal_train.size(0)
                correct = (predicted == sal_train).sum().item()
                precision = 100 * (correct / total)
                print(f'Epoca:{epoch}, Step:{i}, Perdida: {loss.sum()}, Precision: {precision}')
        model.eval()
        evaluate_model(model, testloader, 300, cuda, False)

    print("Entrenamiento finalizado. Evaluacion:")
    evaluate_model(model, testloader, 500, cuda, True)
    print('Si fuese aleatorio:')
    evaluate_random(testloader, 500, False)
    print("Evaluacion finalizada")
    return model


def retrain_save_list_models(list_models, trainloader, testloader, epoch, cuda, path):
    list_models_load = []
    # Cargamos los modelos
    print('Cargando modelos...')
    for name, pretrained in list_models:
        model = AdaptativeNet(pretrained)
        model.load_state_dict(torch.load(path + '/' + name + '.pth'))
        list_models_load.append((name, model))
    print('Modelos cargados, reentrenando...')
    for name, net in list_models_load:
        print('Entrenamiento para: ' + name)
        for param in net.pretrained.parameters():
            param.requires_grad = False
        for param in net.conv1d_layers.parameters():
            param.requires_grad = False
        parameters = []
        parameters.extend(net.conv1d_layers.parameters())
        parameters.extend(net.my_layers.parameters())
        parameters.extend(net.pretrained.parameters())

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(parameters, lr=0.00005)

        if cuda:
            net.to('cuda')
            net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=epoch, cuda=True)
        else:
            net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=epoch, cuda=False)

        torch.save(net.state_dict(), path + '/' + name + '.pth')


def train_save_list_models(list_models, trainloader, testloader, epoch, cuda, path):
    for name, pretrained in list_models:
        print('Entrenamiento para: ' + name)
        net = AdaptativeNet(pretrained)
        for param in net.pretrained.parameters():
            param.requires_grad = False
        # for param in net.conv1d_layers.parameters():
        #    param.requires_grad = False
        parameters = []
        parameters.extend(net.conv1d_layers.parameters())
        parameters.extend(net.my_layers.parameters())
        parameters.extend(net.pretrained.parameters())

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(parameters, lr=0.00005)

        if cuda:
            net.to('cuda')
            net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=epoch, cuda=True)
        else:
            net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=epoch, cuda=False)

        torch.save(net.state_dict(), path + '/' + name + '.pth')


def evaluate_list_models(list_models, testloader, n_times, cuda, path):
    list_models_load = []
    stadistics = []
    combinated_stadistics = []
    combinated_stadistics.append(['Redes combinadas', 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
    # Estadisticas para el test de friedman
    stadistics_fried = []
    combinated_stadistics_fried = []
    combinated_stadistics_fried.append(['Redes combinadas', 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
    evaluations = math.ceil(n_times / 30)
    error_dataframe = {}
    # Cargamos los modelos
    print('Cargando modelos...')
    for name, pretrained in list_models:
        model = AdaptativeNet(pretrained)
        model.load_state_dict(torch.load(path + '/' + name + '.pth'))
        list_models_load.append((name, model))
        stadistics.append([name, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
        stadistics_fried.append([name, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
        error_dataframe[str(name)] = []
    error_dataframe['Combinadas'] = []
    print('Modelos cargados, evaluando...')
    i = 0
    succes = 0
    # Calculamos la salida por cada modelo
    for x_train, y_train, z_train, sal_train in testloader:
        i += 1
        x_train = x_train.float()
        y_train = y_train.float()
        z_train = z_train.float()
        real_exit = sal_train.item()
        if cuda:
            x_train = x_train.to('cuda')
            y_train = y_train.to('cuda')
            z_train = z_train.to('cuda')
        criterio = np.array([0.0, 0.0, 0.0])
        list_results = []
        j = 0
        for name, net in list_models_load:
            net.eval()
            if cuda:
                net.to('cuda')
                output = net(x_train, y_train, z_train)
                output = output.cpu().detach().numpy()
            else:
                output = net(x_train, y_train, z_train)
                output = output.detach().numpy()
            exit_values = ('CERO', 'SUBE', 'BAJA')
            output_number = int(np.argmax(output))
            # Calculamos estadisticas
            stadistics = calculate_stadistic(stadistics, j, i, output_number, real_exit)
            stadistics_fried = calculate_stadistic(stadistics_fried, j, evaluations, output_number, real_exit)
            list_results.append([name, exit_values[output_number]])
            criterio = np.reshape(criterio, (1, 3))
            if i >= 8:
                output = output * stadistics[j][11]
            a = np.concatenate((criterio, output), axis=0)
            criterio = np.sum(a, axis=0)
            j += 1

        # Aplicamos el criterio para elegir la mejor solucion
        result = int(np.argmax(criterio))
        combinated_stadistics = calculate_stadistic(combinated_stadistics, 0, i, result, real_exit)
        combinated_stadistics_fried = calculate_stadistic(combinated_stadistics_fried, 0, evaluations, result,
                                                          real_exit)
        if result == real_exit:
            succes += 1
        # Mostramos los resultados
        print(
            f'Ejecucion:{i}, Prediccion: {exit_values[result]}, Resultado real:{exit_values[real_exit]}, Precision acumulada: {succes / i}')
        print(list_results)
        if i % evaluations == 0:
            z = 0
            new_stadistics = []
            new_combinated_stadistics = []
            new_combinated_stadistics.append(['Redes combinadas', 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
            for name, net in list_models_load:
                error_dataframe[str(name)].append(1 - stadistics_fried[z][11])
                new_stadistics.append([name, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0])
                z = z + 1
            error_dataframe['Combinadas'].append(1 - combinated_stadistics_fried[0][11])
            stadistics_fried = new_stadistics
            combinated_stadistics_fried = new_combinated_stadistics
        if i >= n_times:
            break
    # df = pd.DataFrame(success_dataframe)
    # df.to_csv('/content/drive/MyDrive/TFG/data/estadistica_redes.csv')
    show_stadistics(stadistics)
    print(' Las estadisticas de las redes combinadas serian:')
    show_stadistics(combinated_stadistics)
    vgg = list(error_dataframe.values())[0]
    alexnet = list(error_dataframe.values())[1]
    resnet = list(error_dataframe.values())[2]
    squeezenet = list(error_dataframe.values())[3]
    densenet = list(error_dataframe.values())[4]
    googlenet = list(error_dataframe.values())[5]
    shufflenet = list(error_dataframe.values())[6]
    mobilenet = list(error_dataframe.values())[7]
    resnext = list(error_dataframe.values())[8]
    wide_resnet = list(error_dataframe.values())[9]
    combinated = list(error_dataframe.values())[10]
    f, p, ranking, pivot = friedman_test(vgg, alexnet, resnet, squeezenet, densenet, googlenet, shufflenet, mobilenet,
                                         resnext, wide_resnet, combinated)
    print('ESTADISTICAS DEL TEST DE FRIEDMAN')
    print('F-Value:' + str(f))
    print('P-Value:' + str(p))
    print('Ranking:')
    print(str(list(error_dataframe.keys())))
    print(ranking)
    print('Pivot:')
    print(str(list(error_dataframe.keys())))
    print(pivot)
    w = 0
    pivots = {}
    for name in list(error_dataframe.keys()):
        pivots[str(name)] = list(pivot)[w]
        w += 1
    comparions, z, new_p, adjusted_p = holm_test(pivots, control='Combinadas')
    print('ESTADISTICAS DEL TEST DE HOLM')
    print('Comparions: ' + str(comparions))
    print('P-Values:' + str(new_p))
    print('Adjusted P-Values:' + str(adjusted_p))


def calculate_stadistic(stadistics, j, i, output_number, real_exit):
    if output_number == 0:
        stadistics[j][3] += 1
        if output_number == real_exit:
            stadistics[j][6] += 1
            stadistics[j][10] += 1
            stadistics[j][9] = stadistics[j][6] / stadistics[j][3]
    elif output_number == 1:
        stadistics[j][1] += 1
        if output_number == real_exit:
            stadistics[j][4] += 1
            stadistics[j][10] += 1
            stadistics[j][7] = stadistics[j][4] / stadistics[j][1]
    else:
        stadistics[j][2] += 1
        if output_number == real_exit:
            stadistics[j][5] += 1
            stadistics[j][10] += 1
            stadistics[j][8] = stadistics[j][5] / stadistics[j][2]
    stadistics[j][11] = stadistics[j][10] / i
    return stadistics


def show_stadistics(stadistics):
    dash = '-' * 140
    print(dash)
    print(
        '{:<15s}{:<8s}{:<8s}{:<8s}{:<12s}{:<12s}{:<12s}{:<15s}{:<15s}{:<15s}{:<10s}{:<10s}'.format('Nombre', 'Subidas',
                                                                                                   'Bajadas', 'Ceros',
                                                                                                   'Subidas OK',
                                                                                                   'Bajadas OK',
                                                                                                   'Ceros OK',
                                                                                                   'Porc Subidas',
                                                                                                   'Porc Bajadas',
                                                                                                   'Porc Ceros',
                                                                                                   'Aciertos',
                                                                                                   'Precision'))
    print(dash)
    for stad in stadistics:
        print('{:<15s}{:<8s}{:<8s}{:<8s}{:<12s}{:<12s}{:<12s}{:<15s}{:<15s}{:<15s}{:<10s}{:<10s}'.format(str(stad[0]),
                                                                                                         str(stad[1]),
                                                                                                         str(stad[2]),
                                                                                                         str(stad[3]),
                                                                                                         str(stad[4]),
                                                                                                         str(stad[5]),
                                                                                                         str(stad[6]),
                                                                                                         str(round(
                                                                                                             stad[7],
                                                                                                             4)), str(
                round(stad[8], 4)), str(round(stad[9], 4)), str(stad[10]), str(stad[11])))


def evaluate_model(model, dataloader, n_times, cuda, show=True):
    number = success = ceros = bajadas = subidas = 0
    ceros_real = bajadas_real = subidas_real = 1
    for x_train, y_train, z_train, sal_train in dataloader:
        x_train = x_train.float()
        y_train = y_train.float()
        z_train = z_train.float()
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
            if output_exit == 0:
                ceros += 1
                ceros_real += 1
            elif output_exit == 1:
                subidas += 1
                subidas_real += 1
            else:
                bajadas += 1
                bajadas_real += 1
        else:
            if output_exit == 0:
                ceros_real += 1
            elif output_exit == 1:
                subidas_real += 1
            else:
                bajadas_real += 1
        exit_values = ('CERO', 'SUBE', 'BAJA')
        if show:
            print(
                f'Ejecucion:{number}, Prediccion:{exit_values[int(output_exit)]}, Resultado real:{exit_values[real_exit]}, Precision acumulada:{success / (number + 1)}')
        number += 1
        if number >= n_times:
            break
    precision = success / n_times
    print(f'Evaluacion finalizada, precision: {precision}')
    print(f'CEROS acertados: {ceros}, CEROS totales {ceros_real}, porcentaje: {ceros / ceros_real}')
    print(f'SUBIDAS acertados: {subidas}, SUBIDAS totales {subidas_real}, porcentaje: {subidas / subidas_real}')
    print(f'BAJADAS acertados: {bajadas}, BAJADAS totales {bajadas_real}, porcentaje: {bajadas / bajadas_real}')


def evaluate_random(dataloader, n_times, show=True):
    number = success = ceros = bajadas = subidas = ceros_real = bajadas_real = subidas_real = 0
    for x_train, y_train, z_train, sal_train in dataloader:
        output_exit = randint(0, 2)
        real_exit = sal_train.item()
        if output_exit == real_exit:
            success += 1
            if output_exit == 0:
                ceros += 1
                ceros_real += 1
            elif output_exit == 1:
                subidas += 1
                subidas_real += 1
            else:
                bajadas += 1
                bajadas_real += 1
        else:
            if output_exit == 0:
                ceros_real += 1
            elif output_exit == 1:
                subidas_real += 1
            else:
                bajadas_real += 1
        exit_values = ('CERO', 'SUBE', 'BAJA')
        if show:
            print(
                f'Ejecucion:{number}, Prediccion:{exit_values[int(output_exit)]}, Resultado real:{exit_values[real_exit]}, Precision acumulada:{success / (number + 1)}')
        number += 1
        if number >= n_times:
            break
    precision = success / n_times
    print(f'Evaluacion finalizada, precision: {precision}')
    print(f'CEROS acertados: {ceros}, CEROS totales {ceros_real}, porcentaje: {ceros / ceros_real}')
    print(f'SUBIDAS acertados: {subidas}, SUBIDAS totales {subidas_real}, porcentaje: {subidas / subidas_real}')
    print(f'BAJADAS acertados: {bajadas}, BAJADAS totales {bajadas_real}, porcentaje: {bajadas / bajadas_real}')


def predict(list_models_load, loader, cuda):
    exit_values = ('CERO', 'SUBE', 'BAJA')
    x_train, y_train, z_train = next(iter(loader))
    x_train = x_train.float()
    y_train = y_train.float()
    z_train = z_train.float()
    if cuda:
        x_train = x_train.to('cuda')
        y_train = y_train.to('cuda')
        z_train = z_train.to('cuda')
    criterio = np.array([0.0, 0.0, 0.0])
    list_results = []
    j = 0
    for name, net in list_models_load:
        net.eval()
        if cuda:
            net.to('cuda')
            output = net(x_train, y_train, z_train)
            output = output.cpu().detach().numpy()
        else:
            output = net(x_train, y_train, z_train)
            output = output.detach().numpy()
        output_number = int(np.argmax(output))
        list_results.append([name, exit_values[output_number]])
        criterio = np.reshape(criterio, (1, 3))
        a = np.concatenate((criterio, output), axis=0)
        criterio = np.sum(a, axis=0)
        j += 1
    # Aplicamos el criterio para elegir la mejor solucion
    result = exit_values[int(np.argmax(criterio))]
    # Mostramos los resultados
    return result
