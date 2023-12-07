# Import Libraries
import yfinance as yf
import talib
import matplotlib.pyplot as plt

# Function to Collect Data
def collect_data(stock, start_date, end_date):
    return yf.download(stock, start=start_date, end=end_date)

# 50-Day Moving Average
# Explanation: The 50-day moving average (MA) is a commonly used indicator in technical analysis that represents the average closing price of a security over the last 50 days. It's used to smooth out price data to identify the direction of the trend.
# Importance: It helps traders and analysts to:
# Identify trends: A rising MA suggests an uptrend, while a falling MA indicates a downtrend.
# Determine support and resistance levels: Prices often find support or resistance at the MA.
# Spot potential reversals: Crossovers between the 50-day MA and the price or other moving averages can signal a change in trend.
# Function for 50-Day Moving Average

def calculate_50day_ma(df):
    df['50_MA'] = talib.SMA(df['Close'], timeperiod=50)
    return df

# Relative Strength Index
# Explanation: The RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100.
# Importance:
# Identifies overbought (>70) or oversold (<30) conditions.
# Can indicate potential reversals when divergence occurs between RSI and price.
# Function for Relative Strength Index (RSI)
def calculate_rsi(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    return df

# Bollinger Bands
# Explanation: Bollinger Bands consist of a middle band (20-day SMA) and two outer bands (standard deviations away from the SMA).
# Importance:
# Identifies periods of high or low volatility.
# Prices touching the upper band can indicate overbought conditions, while touching the lower band indicates oversold conditions.
# Function for Bollinger Bands
def calculate_bollinger_bands(df):
    upper_band, middle_band, lower_band = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['Upper_Band'] = upper_band
    df['Middle_Band'] = middle_band
    df['Lower_Band'] = lower_band
    return df

# Moving Average Convergence Divergence (MACD)
# Explanation: MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
# Importance:
# Used to spot changes in the strength, direction, momentum, and duration of a trend.
# Bullish or bearish momentum is indicated by the MACD line crossing above or below the signal line.
# Function for Moving Average Convergence Divergence (MACD)
def calculate_macd(df):
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    return df


def plot_graphs(df, indicator):
    plt.figure(figsize=(10,6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df[indicator], label=f'{indicator}')
    plt.title(f'SPY Close Price and {indicator}')
    plt.legend()
    plt.show()

def main():
    spy_data = collect_data('SPY', '2022-01-01', '2023-11-17')

    # Uncomment the desired function calls
    # spy_data_with_ma = calculate_50day_ma(spy_data)
    # plot_graphs(spy_data_with_ma, '50_MA')

    # spy_data_with_rsi = calculate_rsi(spy_data)
    # plot_graphs(spy_data_with_rsi, 'RSI')

    # spy_data_with_bollinger = calculate_bollinger_bands(spy_data)
    # plot_graphs(spy_data_with_bollinger, 'Upper_Band') # Plot for one of the bands or adjust to plot all

    spy_data_with_macd = calculate_macd(spy_data)
    plot_graphs(spy_data_with_macd, 'MACD')

# Run Script
if __name__ == "__main__":
    main()