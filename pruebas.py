import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from false_dataset import False_dataset
from nets import Alex_net
from net_dataset import Net_dataset
from net_utils import train_model, evaluate_model


pretrained = torchvision.models.alexnet(pretrained=True)
net = Alex_net(pretrained)

net_dataset = Net_dataset("data/ETH-USD-5m.csv", 150)
trainloader = torch.utils.data.DataLoader(net_dataset, batch_size=1, shuffle=False, num_workers=10)
testloader = torch.utils.data.DataLoader(net_dataset, shuffle=True, num_workers=10)


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

if torch.cuda.is_available():
  net.to('cuda')
  net = train_model(net, trainloader, optimizer, criterion, epoch=12, cuda=True)
else:
  net = train_model(net, trainloader, optimizer, criterion, epoch=3, cuda=False)

torch.save(net.state_dict(), 'trained_nets/300-Alex_net_Clasifier.pth')

# Evaluation
net.eval()
evaluate_model(net, testloader, 200, cuda=True)
