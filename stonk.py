import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima

# Download stock data using yfinance
def load_data(ticker):
    data = yf.download(ticker, start="2023-01-01", end=pd.to_datetime('today'))
    data.reset_index(inplace=True)
    return data

# Plot raw stock data
def plot_raw_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Open'], label='Stock Open', color='blue')
    plt.plot(data['Date'], data['Close'], label='Stock Close', color='red')
    plt.title('Raw Stock Data')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show(block=False)

# Forecast using pmdarima
def arima_forecast(data, n_years=1):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    model = auto_arima(df_train['y'], suppress_warnings=True, seasonal=True)
    forecast, conf_int = model.predict(n_periods=n_years*365, return_conf_int=True)

    future_dates = pd.date_range(start=df_train['ds'].max(), periods=n_years*365+1, freq='B')[1:]

    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast, 'yhat_lower': conf_int[:, 0], 'yhat_upper': conf_int[:, 1]})

    return forecast_df

# Main script
user_ticker = input("Enter the stock symbol (e.g., AAPL): ").upper()
n_years = 1

# Load data
stock_data = load_data(user_ticker)

# Set display options for stock_data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display raw data
print(f"Raw Stock Data for {user_ticker}:")
print(stock_data.tail())

# Reset display options to default
pd.set_option('display.max_rows', 10)  # Set the desired maximum number of rows 

# Plot raw data
plot_raw_data(stock_data)

# Perform forecast using pmdarima
forecast = arima_forecast(stock_data, n_years)


# Set display options for forecast
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display forecast data
print("\nForecast Data:")
print(forecast.tail())

# Reset display options to default
pd.set_option('display.max_rows', 10)  # Set the desired maximum number of rows

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Observed', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
plt.title(f'Stock Price Forecast for {n_years} Year(s) using ARIMA - {user_ticker}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
