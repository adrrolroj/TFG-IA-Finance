import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from list_pretrained_nets import get_list_models
from net_dataset_ver2 import Net_dataset_ver2
from net_dataset import Net_dataset, get_train_test_data
from net_utils import train_model, evaluate_model, evaluate_random, train_save_list_models, evaluate_list_models

dash = '-' * 140
print(dash)
print('{:<15s}{:<8s}{:<8s}{:<8s}{:<12s}{:<12s}{:<12s}{:<15s}{:<15s}{:<15s}{:<10s}{:<10s}'.format('Nombre', 'Subidas', 'Bajadas', 'Ceros', 'Subidas OK', 'Bajadas OK', 'Ceros OK', 'Porc Subidas', 'Porc Bajadas', 'Porc Ceros', 'Aciertos', 'Precision'))
print(dash)
'''
list = get_list_models()

net_dataset = Net_dataset_ver2("data/ETH-USD-5m.csv", 150)
train_set, test_set = get_train_test_data(net_dataset, 0.94, 43)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=10)

evaluate_list_models(list, testloader, 200, False, 'trained_nets')
#train_save_list_models(list, trainloader, testloader, 1, False, 'trained_nets')

#net.load_state_dict(torch.load('/content/drive/MyDrive/TFG/trained_nets/300-Alex_net_Clasifier_2.pth'))

# Training
for param in net.pretrained.parameters():
    param.requires_grad = False
for param in net.conv1d_layers.parameters():
    param.requires_grad = False

parameters = []
parameters.extend(net.conv1d_layers.parameters())
parameters.extend(net.my_layers.parameters())
parameters.extend(net.pretrained.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(parameters, lr=0.00001)

if False:
  net.to('cuda')
  net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=12, cuda=True)
else:
  net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=3, cuda=False)

torch.save(net.state_dict(), 'trained_nets/300-Alex_net_Clasifier.pth')

# Evaluation
net.eval()
evaluate_random(testloader, 500, False)
'''