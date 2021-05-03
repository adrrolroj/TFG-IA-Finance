import re
import inquirer
import torch

from finance_utils import update_all_data
from graphic_utils import show_graphic_from_finance, show_graphic_from_finance_and_predict
from net_dataset import Net_dataset, get_train_test_data
from list_pretrained_nets import get_list_models
from net_utils import train_save_list_models, evaluate_list_models

print('¡Bienvenido!')
print('¿Que desea hacer?')
print(' [1] Actualizar la base de datos de todos los mercados \n [2] Ver graficas de mercado a tiempo real \n [3] Reentrenar las redes neuronales para Etherium \n [4] Evaluar las redes neuronales \n [5] Ver graficas de etherium a tiempo real con prediccion')
option = input('Seleccione opcion: ')
if option == '1':
    update_all_data()
    print("Todos los datos han sido actualizados con exito")
elif option == '2':
    with open('markets.txt') as f:
        lines = f.readlines()
    book = {}
    for line in lines:
        book[line.split("==")[1].rstrip("\n")] = line.split("==")[0]
    i = 0
    print('¿Que mercado quiere visualizar?')
    for key in book.keys():
        i += 1
        print('[' + str(i) + '] ' + key)
    market = input('Seleccione:')
    market = list(book.values())[int(market)-1]
    file = "data/" + market + "-5m.csv"
    number = input('¿Cuantas velas quiere visualizar?: ')
    show_graphic_from_finance(file, int(number), market)
elif option == '3':
    print('Se reentrenaran todas las redes disponibles en list_pretrained_nets.py')
    cuda = input('¿Quieres usar CUDA?(SI/NO): ')
    batch = input('¿Tamaño batch?: ')
    epoch = input('¿Numero de epocas?: ')
    net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
    train_set, test_set = get_train_test_data(net_dataset, 0.94, 42)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=int(batch), shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=2)

    list_nets = []
    for net in get_list_models():
        list_nets.append((net[0], net[1]))
    if cuda == 'SI' and torch.cuda.is_available:
        train_save_list_models(list_nets, trainloader, testloader, int(epoch), True, 'trained_nets')
    elif cuda == 'NO':
        print('No se va a usar cuda')
        train_save_list_models(list_nets, trainloader, testloader, int(epoch), False, 'trained_nets')
    else:
        print('No se ha entendido si quiere suar cuda o no, repitalo de nuevo')

elif option == '4':
    print('Se evaluaran todas las redes disponibles en list_pretrained_nets.py')
    cuda = input('¿Quieres usar CUDA?(SI/NO): ')
    eval = input('Numero de evaluaciones: ')

    net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
    train_set, test_set = get_train_test_data(net_dataset, 0.94, 39)

    testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=2)

    list_nets = []
    for net in get_list_models():
        list_nets.append((net[0], net[1]))
    if cuda == 'SI' and torch.cuda.is_available:
        with torch.no_grad():
            evaluate_list_models(list_nets, testloader, int(eval), True, 'trained_nets')
    elif cuda == 'NO':
        print('No se utilizara CUDA')
        with torch.no_grad():
            evaluate_list_models(list_nets, testloader, int(eval), False, 'trained_nets')
    else:
        print('No se ha entendido si quiere suar cuda o no, repitalo de nuevo')

elif option == '5':
    cuda = input('¿Quieres usar CUDA?(SI/NO): ')
    if cuda == 'SI' and torch.cuda.is_available:
        show_graphic_from_finance_and_predict(True)
    elif cuda == 'NO':
        print('No se utilizara CUDA')
        show_graphic_from_finance_and_predict(False)
    else:
        print('No se ha entendido si quiere suar cuda o no, repitalo de nuevo')
else:
    print('Seleccion invalida, intentelo de nuevo')

