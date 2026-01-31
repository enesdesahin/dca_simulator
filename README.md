# DCA Simulator

A powerful, interactive Streamlit application to calculate and visualize Dollar Cost Averaging (DCA) investment strategies.

## Features

- **Real-Time Data**: Fetches historical market data for stocks, ETFs, and cryptocurrencies using Yahoo Finance.
- **DCA Simulation**: Invest a fixed monthly amount over any historical period.
- **Multi-Currency Support**: Automatically handles currency conversion (USD, EUR, GBP, CHF) for accurate portfolio valuation in your local currency.
- **Interactive Visualization**: Dynamic Plotly charts comparing "Portfolio Value" vs. "Total Invested" over time.
- **Animation Mode**: "Replay" the investment journey with a smooth animation loop.
- **Export Data**: Download the simulation results as a CSV file.
- **Mobile Optimized**: Responsive design that looks great on both desktop and mobile devices.

## Installation

1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd dca_simulator
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

Run the Streamlit application:

```bash
streamlit run dca_app.py
```

The app will open in your default browser (usually at `http://localhost:8501`).

## Configuration

Use the sidebar to configure your simulation:
- **Select Asset**: Choose from popular presets (SPY, BTC-USD, NVDA...) or enter any Yahoo Finance ticker.
- **Date Range**: Pick the start and end date for your backtest.
- **Monthly Investment**: Set the amount you want to "invest" each month.
- **Currency**: View values in USD, EUR, GBP, or CHF (handles historical exchange rates automatically).

## Built With

- [Streamlit](https://streamlit.io/)
- [Yahoo Finance (yfinance)](https://github.com/ranaroussi/yfinance)
- [Plotly](https://plotly.com/)
- [Pandas](https://pandas.pydata.org/)
