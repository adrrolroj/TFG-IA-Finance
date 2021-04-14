import re
import inquirer
import torch

from finance_utils import update_all_data
from graphic_utils import show_graphic_from_finance, show_graphic_from_finance_and_predict
from net_dataset import Net_dataset, get_train_test_data
from list_pretrained_nets import get_list_models
from net_utils import train_save_list_models, evaluate_list_models

print('¡Bienvenido!')
options = [
    inquirer.List(
        "Option",
        message="¿Que desea hacer?",
        choices=["Actualizar la base de datos de todos los mercados", "Ver graficas de mercado a tiempo real",
                 "Reentrenar las redes neuronales para Etherium", "Evaluar las redes neuronales", "Ver graficas de etherium a tiempo real con prediccion"],
    ),
]

answer = inquirer.prompt(options)
if answer["Option"] == "Ver graficas de mercado a tiempo real":
    with open('markets.txt') as f:
        lines = f.readlines()
    book = {}
    for line in lines:
        book[line.split("==")[1].rstrip("\n")] = line.split("==")[0]
    markets = [
        inquirer.List(
            "market",
            message="¿Que mercado desea ver?",
            choices=book.keys(),
        ),
        inquirer.Text('number', message="¿Cuantas velas deseeas visualizar?",
                      validate=lambda _, x: re.match('^[1-9][0-9]*$', x)),
    ]
    answer = inquirer.prompt(markets)
    file = "data/" + book.get(answer["market"]) + "-5m.csv"
    show_graphic_from_finance(file, int(answer["number"]), book.get(answer["market"]))
elif answer["Option"] == "Reentrenar las redes neuronales para Etherium":
    question_nets = [
        inquirer.Checkbox('nets',
                          message="¿Que redes deseas entrenar?",
                          choices=['vgg16', 'alexnet', 'resnet18', 'squeezenet', 'densenet', 'googlenet', 'shufflenet',
                                   'mobilenet', 'resnext50', 'wide_resnet'],
                          ),
        inquirer.List(
            "Cuda",
            message="¿Quieres usar CUDA?",
            choices=["SI", "NO"],
        ),
        inquirer.Text('batch', message="¿Tamaño del batch?",
                      validate=lambda _, x: re.match('^[1-9][0-9]*$', x)),
        inquirer.Text('epoch', message="¿Numero de epocas?",
                      validate=lambda _, x: re.match('^[1-9][0-9]*$', x)),
    ]
    answers_nets = inquirer.prompt(question_nets)
    net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
    train_set, test_set = get_train_test_data(net_dataset, 0.94, 42)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=int(answers_nets["batch"]), shuffle=True,
                                              num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=2)

    list_nets = []
    list_select = answers_nets["nets"]
    for net in get_list_models():
        if net[0] in list_select:
            list_nets.append((net[0], net[1]))
    if answers_nets["Cuda"] == 'SI' and torch.cuda.is_available:
        train_save_list_models(list_nets, trainloader, testloader, int(answers_nets["epoch"]), True, 'trained_nets')
    else:
        train_save_list_models(list_nets, trainloader, testloader, int(answers_nets["epoch"]), False, 'trained_nets')
elif answer["Option"] == "Evaluar las redes neuronales":
    question_nets = [
        inquirer.Checkbox('nets',
                          message="¿Que redes deseas evaluar?",
                          choices=['vgg16', 'alexnet', 'resnet18', 'squeezenet', 'densenet', 'googlenet', 'shufflenet',
                                   'mobilenet', 'resnext50', 'wide_resnet'],
                          ),
        inquirer.List(
            "Cuda",
            message="¿Quieres usar CUDA?",
            choices=["SI", "NO"],
        ),
        inquirer.Text('eval', message="¿Numero de evaluaciones?",
                      validate=lambda _, x: re.match('^[1-9][0-9]*$', x)),
    ]
    answers_nets = inquirer.prompt(question_nets)
    net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
    train_set, test_set = get_train_test_data(net_dataset, 0.94, 42)

    testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=2)

    list_nets = []
    list_select = answers_nets["nets"]
    for net in get_list_models():
        if net[0] in list_select:
            list_nets.append((net[0], net[1]))
    if answers_nets["Cuda"] == 'SI' and torch.cuda.is_available:
        with torch.no_grad():
            evaluate_list_models(list_nets, testloader, int(answers_nets["eval"]), True, 'trained_nets')
    else:
        with torch.no_grad():
            evaluate_list_models(list_nets, testloader, int(answers_nets["eval"]), False, 'trained_nets')
elif answer["Option"] == "Ver graficas de etherium a tiempo real con prediccion":
    question_nets = [
        inquirer.List(
            "Cuda",
            message="¿Quieres usar CUDA?",
            choices=["SI", "NO"],
        ),
    ]
    answers_nets = inquirer.prompt(question_nets)
    if answers_nets["Cuda"] == 'SI' and torch.cuda.is_available:
        show_graphic_from_finance_and_predict(True)
    else:
        show_graphic_from_finance_and_predict(False)
else:
    update_all_data()
    print("Todos los datos han sido actualizados con exito")
