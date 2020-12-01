import math
import os

import yfinance as yf
import pandas as pd
from datetime import date, timedelta


def save_finance_in_csv(moneda, archivo, intervalo):
    data = get_data_from_finance(moneda, intervalo)
    data = check_continuity_data(data)
    data.to_csv(archivo, index=True)
    return data


def get_data_from_finance(moneda, intervalo):
    # Día actual
    today = date.today()
    i = 5
    if intervalo == "5m":
        i = 11
    data = yf.download(tickers=moneda, start=str(today - timedelta(days=4)),
                       interval=intervalo)
    for day in range(i):
        start = str(today - timedelta(days=((day + 2) * 5) - 1))
        end = str(today - timedelta(days=((day + 1) * 5) - 1))
        dataConcat = yf.download(tickers=moneda, start=start, end=end,
                                 interval=intervalo)
        data = pd.concat([data, dataConcat])
        data = data[~data.index.duplicated(keep='last')]

    data = data.sort_index()
    return data


def update_data_from_finance(archivo, moneda, intervalo):
    # Día actual
    today = date.today()
    df = pd.read_csv(archivo, index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.drop(df.index.max())
    recent_date = df.index.max()
    difference = today - recent_date.date()
    data = pd.DataFrame()
    if difference.days >= 60:
        print("Han pasado mas de 60 dias desde la ultima actualizacion")
        recent_date = recent_date.date() - timedelta(days=(difference.days - (difference.days - 59)))
        difference = today - recent_date

    if difference.days > 5:
        i = math.ceil((difference.days / 5) - 1)
        data = yf.download(tickers=moneda, start=str(today - timedelta(days=4)),
                           interval=intervalo)
        for day in range(i):
            start = str(today - timedelta(days=((day + 2) * 5) - 1))
            end = str(today - timedelta(days=((day + 1) * 5) - 1))
            dataConcat = yf.download(tickers=moneda, start=start, end=end,
                                     interval=intervalo)
            data = pd.concat([data, dataConcat])
    else:
        data = yf.download(tickers=moneda, start=recent_date,
                           interval=intervalo)

    data.index = pd.to_datetime(data.index, utc=True)
    data = pd.concat([df, data])
    data = data[~data.index.duplicated(keep='last')]
    data = check_continuity_data(data)
    data = data.sort_index()

    data.to_csv(archivo, index=True)
    return data


def check_continuity_data(dataframe):
    print("Checking continuity...")
    recent_date = pd.to_datetime(dataframe.index.max())
    actual_date = pd.to_datetime(dataframe.index.min())
    while actual_date < recent_date:
        if not actual_date in dataframe.index:
            previous = pd.to_datetime(actual_date, utc=True) - timedelta(minutes=5)
            new_data = dataframe.loc[previous, :]
            close_price = new_data['Close']
            datas = {'Datetime': [actual_date],
                     'Open': [close_price],
                     'High': [close_price],
                     'Low': [close_price],
                     'Close': [close_price],
                     'Volume': [0],
                     'Adj Close': [close_price],
                     }

            append = pd.DataFrame(datas)
            append = append.set_index('Datetime')
            dataframe = dataframe.append(append)
        actual_date = pd.to_datetime(actual_date, utc=True) + timedelta(minutes=5)
    dataframe = dataframe.sort_index()
    return dataframe


def update_all_data():
    with open('markets.txt') as f:
        lines = f.readlines()
    for line in lines:
        print(line.split("==")[1].rstrip("\n"))
        mercado = line.split("==")[0]
        archivo = "data/" + line.split("==")[0] + "-5m.csv"
        if os.path.exists(archivo):
            update_data_from_finance(archivo, mercado, "5m")
        else:
            save_finance_in_csv(mercado, archivo, "5m")

