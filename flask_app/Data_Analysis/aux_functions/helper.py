import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from aux_functions.plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10

####################################
# Objetivo: Regressão linear SP500 #
####################################
#Monthly_investment = 500
#simulacoes = 10000
#Future_Years = 5

def future_brownian(sigma, avg_expo, stock_data, years, simulacoes):

    last_price = float(stock_data['Close'].iloc[-1])

    expected_return = (avg_expo[-1] / last_price)**(1/(12*years))
    mu = np.log(expected_return)  # anualizado

    # Data final dos teus dados reais
    last_date = stock_data['Date'].max()

    # Gerar datas mensais futuras
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods= 12 * years, freq='MS')


    rand_norm = np.random.normal(size=(12*years-1, simulacoes))
    drift = mu - 0.5 * sigma**2
    steps = np.exp(drift + sigma * rand_norm)

    # Inicializar matriz de preços
    precos = np.zeros((12*years, simulacoes))
    precos[0] = last_price
    pd.DataFrame(precos)

    # Corrigir multiplicação
    precos[1:, :] = last_price * np.array(steps.cumprod(axis=0))

    # Gerar DataFrame
    precos_df = pd.DataFrame(precos, index=future_dates)


    #plot7(precos_df, future_dates, avg_expo)
    return precos_df, future_dates
# %%
#future_brownian(sigma, y_pred_future, sp500_data, Future_Years, simulacoes)