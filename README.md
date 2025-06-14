# 📈 S&P 500 Monte Carlo Investment Simulator (Flask App)

This is a web application built with **Flask** that simulates investment strategies on the S&P 500 using **dynamic allocation**, **exponential regression**, and **Monte Carlo simulations**. Users can visualize performance compared to Buy & Hold and explore optimal investment timing based on historical data.

---

## 🚀 Features

- 📊 **S&P 500 Data Analysis** with exponential regression fitting
- 💸 **Monthly Investment Strategy** with dynamic allocation
- 🎲 **Monte Carlo Simulations** with randomized start dates
- 📉 **Backtesting Module** with ROI comparison
- 🌐 **Flask Web App** for user interaction and visualization
- 📁 **SQLite database** to store simulation results
- 📄 Clean web templates (Jinja2)

---

## 🗂️ Project Structure
S-P-500-MONTE-CARLO-CALCULATOR/
│
├── flask_app/
│ ├── Data_Analysis/
│ │ ├── aux_functions/
│ │ ├── S&P500_Data/
│ │ └── backtest_and_montecarlo.py
│ ├── flask_session/
│ ├── help/
│ └── static/
│
├── templates/
│ ├── index.html
│ ├── apology.html
│ └── layout.html
│
├── app.py # Main Flask application
├── helpers.py # Custom helper functions
├── finance.db # SQLite3 database file
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md


---

## 📌 Setup Instructions

1. Clone the Repository

    ```bash
    git clone https://github.com/yourusername/S-P-500-MONTE-CARLO-CALCULATOR.git
    cd S-P-500-MONTE-CARLO-CALCULATOR

2. Create and Activate Virtual Environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install Dependencies
    pip install -r requirements.txt

4. Run the App
    flask run
    Then open http://127.0.0.1:5000 in your browser.

## 🧠 Strategy Overview

Uses exponential regression to estimate the fair value of the S&P 500 index.

Allocates monthly investments based on how overvalued or undervalued the index is.

Implements Monte Carlo simulations by randomizing investment starting dates.

Compares performance against a standard Buy and Hold approach.


## 📚 Technologies Used

Python 3

Flask – web framework

Jinja2 – HTML templating

SQLite – database

pandas, numpy, matplotlib, seaborn, yfinance

## 🧾 License

This project is licensed under the MIT License.
Feel free to modify and reuse for educational or personal use.

## 🙋 Author
Developed by Pedro Maltez.
For questions, suggestions, or contributions, feel free to open an issue or contact via GitHub.


Disclaimer: This tool is for educational purposes only and does not constitute financial advice.
