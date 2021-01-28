import torch.nn as nn
import torch
from torchvision.transforms import transforms

'''
PyTorch necesita que en las operaciones convolucionales las imagen vengan en formato 
(tamaño_batch, n_canales, altura, anchura), la primera operación del forward se encarga
 de cambiar la forma del tensor a ese formato pero no intercambia los valores, en el caso
  de tener una imgen con formato (tamaño_batch, altura, anchura, n_canales)
'''


class Alex_net(nn.Module):

    def __init__(self, pretrained_model):
        super(Alex_net, self).__init__()
        self.conv1d_layers = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3),
                                           nn.Conv1d(32, 128, 5, padding=2),
                                           nn.ReLU())
        self.pretrained = pretrained_model
        self.my_layers = nn.Sequential(nn.Dropout(p=0.5),
                                       nn.Linear(1000, 800),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(800, 500),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(500, 200),
                                       nn.Linear(200, 3))

    def forward(self, x, y, z):
        # Concatenamos para tener los 3 canales RGB
        x = x.view(1, 1, x.shape[0])
        y = y.view(1, 1, y.shape[0])
        z = z.view(1, 1, z.shape[0])
        x = torch.cat((y, x, z), 0)

        x = self.conv1d_layers(x)
        x = x.view(-1, 3, x.shape[1], x.shape[2])
        x = self.pretrained(x)
        x = self.my_layers(x)
        return x

