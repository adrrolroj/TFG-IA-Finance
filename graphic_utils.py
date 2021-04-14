from datetime import datetime

import pandas as pd
import mplfinance as mpf
import matplotlib.animation as animation
import torch

from finance_utils import update_data_from_finance
from list_pretrained_nets import get_list_models
from net_dataset_for_show import Net_dataset_show
from net_utils import predict
from nets import AdaptativeNet


def show_graphic_from_finance(archivo, numero_velas, moneda):
    df = pd.read_csv(archivo, index_col='Datetime')

    df.index = pd.to_datetime(df.index, utc=True)
    fig = mpf.figure(style='charles', figsize=(7, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(3, 1, 3)

    def animate(ival):
        data = update_data_from_finance(archivo, moneda, "5m")
        print("UPDATED")
        print(data.index.max())
        data_to_print = data.iloc[(len(data) - numero_velas):len(data)]
        ax1.clear()
        ax2.clear()
        mpf.plot(data_to_print, ax=ax1, volume=ax2, type='candle')

    ani = animation.FuncAnimation(fig, animate, interval=300000)
    mpf.show()


def show_graphic_from_finance_and_predict(cuda):
    df = pd.read_csv("data/ETH-USD-5m.csv", index_col='Datetime')

    df.index = pd.to_datetime(df.index, utc=True)
    fig = mpf.figure(style='charles', figsize=(7, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(3, 1, 3)

    list_models_load = []
    list_models = get_list_models()
    # Cargamos los modelos
    print('Cargando modelos...')
    for name, pretrained in list_models:
        model = AdaptativeNet(pretrained)
        model.load_state_dict(torch.load('trained_nets/' + name + '.pth'))
        list_models_load.append((name, model))
    print('Modelos cargados')

    def animate(ival):
        data = update_data_from_finance("data/ETH-USD-5m.csv", "ETH-USD", "5m")
        print("UPDATED")
        print(data.index.max())
        data_to_print = data.iloc[(len(data) - 150):len(data)]

        net_dataset = Net_dataset_show(data_to_print)
        loader = torch.utils.data.DataLoader(net_dataset, shuffle=True, num_workers=2)

        with torch.no_grad():
            prediction = predict(list_models_load, loader, cuda)
        print('La prediccion de la proxima vela sera: ' + prediction)
        ax1.clear()
        ax2.clear()
        mpf.plot(data_to_print, ax=ax1, volume=ax2, type='candle')

    ani = animation.FuncAnimation(fig, animate, interval=300000)
    mpf.show()
