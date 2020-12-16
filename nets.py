import torch.nn as nn
import torch

'''
PyTorch necesita que en las operaciones convolucionales las imagen vengan en formato 
(tamaño_batch, n_canales, altura, anchura), la primera operación del forward se encarga
 de cambiar la forma del tensor a ese formato pero no intercambia los valores, en el caso
  de tener una imgen con formato (tamaño_batch, altura, anchura, n_canales)
'''


class Alex_net(nn.Module):

    def __init__(self, pretrained_model, size_exit):
        super(Alex_net, self).__init__()
        self.conv1d_layers_x = nn.Sequential(nn.Conv1d(1, 16, 5, stride=2, padding=2, padding_mode='reflect'),
                                             nn.ReLU(),
                                             nn.Conv1d(16, 128, 3, stride=2, padding=1, padding_mode='reflect'),
                                             )
        self.conv1d_layers_y = nn.Sequential(nn.Conv1d(1, 16, 5, stride=2, padding=2, padding_mode='reflect'),
                                             nn.ReLU(),
                                             nn.Conv1d(16, 128, 3, padding=1, padding_mode='reflect'),
                                             )
        self.conv1d_layers_z = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2, padding_mode='reflect'),
                                             nn.ReLU(),
                                             nn.Conv1d(16, 128, 3, padding=1, padding_mode='reflect'),
                                             )
        self.pretrained = pretrained_model
        self.my_layers = nn.Sequential(nn.Dropout(p=0.1),
                                       nn.Linear(1000, 1000),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(1000, 500),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(500, 200),
                                       nn.ReLU(),
                                       nn.Linear(200, size_exit))

    def forward(self, x, y, z):
        # Se ejecutan la 3 conv1d para 5 min, 10 min, y 20 min
        x = x.view(1, 1, x.shape[0])
        y = y.view(1, 1, y.shape[0])
        z = z.view(1, 1, z.shape[0])
        x = self.conv1d_layers_x(x)
        y = self.conv1d_layers_y(y)
        z = self.conv1d_layers_z(z)
        # El resultado se concatena para dar lugar a una imagen RGB
        x = torch.cat((x, y, z), 0)
        x = x.view(-1, 3, x.shape[1], x.shape[2])
        x = self.pretrained(x)
        x = self.my_layers(x)
        return x
