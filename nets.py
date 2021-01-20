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
        self.conv1d_layers = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2),
                                           nn.Conv1d(32, 128, 5),
                                           nn.ReLU(inplace=True),
                                           nn.Conv1d(128, 256, 3)
                                           )
        self.pretrained = pretrained_model
        self.my_layers = nn.Sequential(nn.Dropout(p=0.3),
                                       nn.Linear(1000, 1000),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(1000, 500),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(500, 200),
                                       nn.ReLU(),
                                       nn.Linear(200, 3))

    def forward(self, x, y, z):
        # Se ejecutan la 3 conv1d para 5 min, 10 min, y 20 min
        x = x.view(1, 1, x.shape[0])
        y = y.view(1, 1, y.shape[0])
        z = z.view(1, 1, z.shape[0])
        x = self.conv1d_layers(x)
        y = self.conv1d_layers(y)
        z = self.conv1d_layers(z)
        # El resultado se concatena para dar lugar a una imagen RGB
        x = torch.cat((x, y, z), 0)
        #results2 = transforms.ToPILImage()(x).convert("RGB")
        #results2.save('hola.png')
        x = x.view(-1, 3, x.shape[1], x.shape[2])
        x = self.pretrained(x)
        x = self.my_layers(x)
        return x
