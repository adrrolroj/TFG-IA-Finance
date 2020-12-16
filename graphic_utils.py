from datetime import datetime

import pandas as pd
import mplfinance as mpf
import matplotlib.animation as animation
from finance_utils import update_data_from_finance


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

    animation.FuncAnimation(fig, animate, interval=60000)
    mpf.show()
