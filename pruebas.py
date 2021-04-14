import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from list_pretrained_nets import get_list_models
from net_dataset import Net_dataset
from false_dataset import False_dataset
from net_dataset import Net_dataset, get_train_test_data
from net_utils import train_model, evaluate_model, train_save_list_models, evaluate_list_models, retrain_save_list_models


torch.set_printoptions(precision=15)

net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
train_set, test_set = get_train_test_data(net_dataset, 0.94, 42)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=300, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=2)


list_nets = get_list_models()
with torch.no_grad():
    evaluate_list_models(list_nets, testloader, 500, False, 'trained_nets')

#train_save_list_models(list_nets, trainloader, testloader, 12, False, 'trained_nets')
'''
for param in net.pretrained.parameters():
    param.requires_grad = False

parameters = []
parameters.extend(net.conv1d_layers.parameters())
parameters.extend(net.my_layers.parameters())
parameters.extend(net.pretrained.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(parameters, lr=0.00005)

if torch.cuda.is_available():
  net = train_model(net, trainloader, testloader, optimizer, criterion, epoch=15, cuda=True)
else:

torch.save(net.state_dict(), '/content/drive/MyDrive/TFG/trained_nets/150-Alex_net_Clasifier.pth')
'''
