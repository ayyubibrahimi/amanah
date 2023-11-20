import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pywt
import kerastuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_model(hp, input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=hp.Int('units', min_value=50, max_value=150, step=50),
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=50, max_value=150, step=50), return_sequences=False))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def mad(data):
    """
    Median Absolute Deviation: a robust measure of statistical dispersion.
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def wavelet_denoising(data, wavelet='db4', level=1):
    """
    Wavelet denoising function, adjusted to match the length of input data.
    """
    original_length = len(data)
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    denoised_data = pywt.waverec(coeff, wavelet, mode='per')

    # Adjust the length of denoised data to match the original data length
    if len(denoised_data) > original_length:
        return denoised_data[:original_length]
    elif len(denoised_data) < original_length:
        return np.pad(denoised_data, (0, original_length - len(denoised_data)), 'constant', constant_values=(0))
    else:
        return denoised_data

# Data Collection
def collect_data(stock, start_date, end_date):
    return yf.download(stock, start=start_date, end=end_date)


# Feature Engineering with Denoising
def feature_engineering(df):
    # Calculating raw features
    df['Return'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()

    # Denoising
    df['Close'] = wavelet_denoising(df['Close'])
    df['Volume'] = wavelet_denoising(df['Volume'])


    # Recalculating features after denoising
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()

    # Drop NaN values
    df.dropna(inplace=True)

    # Scaling
    scaler = MinMaxScaler()
    df[["MA_7", "MA_20", "MA_50", "Volume_Change"]] = scaler.fit_transform(df[["MA_7", "MA_20", "MA_50", "Volume_Change"]])
    return df


def train_lstm_model(X, y, epochs=100, batch_size=32):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
    return model


# Main Function with LSTM Model Training
def main():
    spy_data = collect_data('SPY', '2023-01-01', '2023-11-19')
    spy_data_fe = feature_engineering(spy_data)

    features = ["MA_7", 'MA_20', 'MA_50', 'Volume_Change']
    X = spy_data_fe[features].values
    y = spy_data_fe['Return'].values

    # Reshape data for LSTM model
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Prepare data for LSTM model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Hyperparameter tuning
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=3,
        directory='my_dir',
        project_name='SPY_LSTM'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=2)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Retrain the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

    # Making Predictions
    predicted_y = model.predict(X_test)

    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predicted_y))
    mae = mean_absolute_error(y_test, predicted_y)

    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')

    # Reshaping y_test and predicted_y for plotting
    y_test = y_test.reshape(-1, 1)
    predicted_y = predicted_y.reshape(-1, 1)

    latest_input = spy_data_fe[features].values[-1].reshape(1, 1, -1)
    tomorrow_prediction = model.predict(latest_input)
    prediction_text = f"Predicted Return for Tomorrow: {tomorrow_prediction[0][0]:.4f}"

    # Plotting the results
    plt.figure(figsize=(10,6))
    plt.plot(y_test, color='blue', label='Actual SPY Returns')
    plt.plot(predicted_y, color='red', linestyle='--', label='Model Predicted SPY Returns')

    # Adding the prediction as text on the plot
    plt.text(0.05, 0.95, prediction_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.title('SPY Stock Returns Prediction')
    plt.xlabel('Time')
    plt.ylabel('Normalized Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()