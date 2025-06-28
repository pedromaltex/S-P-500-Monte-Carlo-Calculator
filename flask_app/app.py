import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from flask import jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from yahooquery import Screener
import numpy as np
import sys
import os

# Caminho absoluto para o diretório onde está 'aux_functions'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Data_Analysis")))

from helpers import apology, login_required, lookup, usd, get_data, get_news, correlation, get_interval, get_data_percent, compare2, dowl_data_return_dataset, calc_returns_daily, calc_corr, get_companiesbysector, heatmap, download_data
from Data_Analysis.backtest_and_montecarlo import monte_carlo


# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

# Sets the time of the events
current_time = datetime.now()


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    #if request.method == "POST":
    # Extract form inputs (with error handling)
    try:
        monthly = float(request.form.get("Monthly", 100))
        syear = int(request.form.get("SYear", 2000))
        fyear = int(request.form.get("FYear", 2025))

        n_simulations = int(request.form.get("NSimulations", 10000))
        future_years = int(request.form.get("future", 5))
    except (ValueError, TypeError):
        return "Invalid input values", 400

    # Run Monte Carlo simulation
    variables = monte_carlo(monthly, syear, fyear, n_simulations, future_years)
    timeline = [dt.strftime('%Y-%m-%d') for dt in variables[0]]

    sp500_prices = variables[1]['^GSPC'].tolist()
    av_log = variables[3].tolist()
    log_sp500 = variables[4]['^GSPC'].tolist()
    av_exp = variables[7].tolist()
    exp_sp = variables[8].tolist()
    diference = variables[9].tolist()
    s_portfolio = variables[10].tolist()
    s_portfolio2 = variables[11].tolist()
    s_total_invest=variables[14].tolist()
    s_total_allocation=variables[15].tolist()
    dif = (variables[11] - variables[10]) # / variables[10] ?
    dif = dif.tolist()

    montecarlo = variables[-1]
    min_values = montecarlo.min(axis=1).tolist()
    max_values = montecarlo.max(axis=1).tolist()
    mean_values = montecarlo.mean(axis=1).tolist()

    future_dates = montecarlo.index.strftime('%Y-%m-%d').tolist()
    y_pred_future = variables[-2].tolist()

    roi_standart = variables[20].tolist()
    roi_maltez = variables[21].tolist()
    bins = variables[22].tolist()
    bin_min_value = variables[23]
    bin_max_value = variables[24]
    bin_width = variables[25]

    final_values1=variables[16]
    final_values2=variables[17]

    # Define os limites globais
    min_val2 = min(final_values1.min(), final_values2.min())
    max_val2 = max(final_values1.max(), final_values2.max())
    bin_width2 = final_values2.max()/100

    # Gera os bins com mesma largura
    bins2 = np.arange(np.floor(min_val2), np.ceil(max_val2) + bin_width2, bin_width2)
    bins2=bins2.tolist()
    final_values1=final_values1.tolist()
    final_values2=final_values2.tolist()
    
    monte_carlo_over_under_valuation = 100*(max_values[0] - y_pred_future[0]) / y_pred_future[0]

    monte_carlo_over_under_valuation_str = f"Undervalued by {monte_carlo_over_under_valuation:.2f}"
    if monte_carlo_over_under_valuation >= 0:
        monte_carlo_over_under_valuation_str = f"Overvalued by {monte_carlo_over_under_valuation:.2f}"
    

    parametros = variables[-3]
    coef_0 = parametros[0]
    coef_1 = parametros[1]
    cagr = parametros[2]
    s_allocated = f"{parametros[3]:.2f}"
    s_port_final = f"{parametros[4]:.2f}"
    s_allocated2 = f"{parametros[5]:.2f}"
    s_port_final2 = f"{parametros[6]:.2f}"
    media_roi_maltez = f"{parametros[7]:.2f}"
    media_roi_std = f"{parametros[8]:.2f}"
    media_std = f"{parametros[9]:.2f}"
    media_maltez = f"{parametros[10]:.2f}"


    # Render the SAME template with results
    return render_template("index.html", 
        plots=True,  # Flag to show plots section
        timeline=timeline, \
        sp500_prices=sp500_prices, \
        name=variables[2], \
        av_log=av_log, \
        log_sp500=log_sp500, \
        coef1=variables[5], \
        coef2=variables[6], \
        av_exp=av_exp, \
        exp_sp=exp_sp, \
        diference=diference, \
        s_portfolio=s_portfolio, \
        s_portfolio2=s_portfolio2, \
        s_total_invest=s_total_invest, \
        s_total_allocation=s_total_allocation, \
        portfolio=variables[12], \
        porfolio2=variables[13], \
        total_invest=variables[12], \
        total_allocation=variables[13], \
        final_values1=final_values1, \
        final_values2=final_values2, \
        total_allocation2=variables[16], \
        final_allocation=variables[17], \
        roi_standart=roi_standart, \
        roi_maltez=roi_maltez, \
        bins=bins,  \
        min_values=min_values, \
        max_values=max_values , future_dates=future_dates, \
        mean_values=mean_values, y_pred_future=y_pred_future, \
        dif=dif, bin_min_value=bin_min_value, \
        bin_max_value=bin_max_value, bin_width=bin_width, \
        bin_min_value2=min_val2, bin_max_value2=max_val2, bins2=bins2, \
        bin_width2=bin_width2, \
        coef_0=coef_0, \
        coef_1=coef_1, \
        cagr=cagr, \
        s_allocated=s_allocated, \
        s_port_final=s_port_final, \
        s_allocated2=s_allocated2, \
        rel_dif_alloc=f"{(100*(float(s_allocated2)-float(s_allocated))/float(s_allocated)):.2f}", \
        rel_dif_inv=f"{100*(float(s_port_final2)-float(s_port_final))/float(s_port_final):.2f}", \
        s_port_final2=s_port_final2, \
        media_roi_maltez=media_roi_maltez, \
        media_roi_std=media_roi_std, \
        media_maltez=media_maltez, \
        media_std=media_std, \
        monte_carlo_over_under_valuation_str=monte_carlo_over_under_valuation_str,

    )

    # GET request: Show empty form
    return render_template("index.html", plots=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
