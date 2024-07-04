from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

app = Flask(__name__)

# Fetch data from Yahoo Finance
def fetch_data_from_yahoo(ticker, period='5y'):
    data = yf.download(ticker, period=period, interval='1d')
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
    data['NetChange'] = data['Close'].diff()
    return data

# Data normalization function
def normalize_data(df):
    sc = MinMaxScaler()
    Featured_Data_Close = df[['Close']].values
    DataScaler_Close = sc.fit(Featured_Data_Close)
    X = DataScaler_Close.transform(Featured_Data_Close)
    return X, DataScaler_Close

# Load LSTM model function
def load_lstm_model(TimeSteps, TotalFeatures):
    model_Close = Sequential([
        Bidirectional(LSTM(units=128, activation='relu', return_sequences=True), input_shape=(TimeSteps, TotalFeatures)),
        Dropout(0.3),
        Bidirectional(LSTM(units=64, activation='relu', return_sequences=True)),
        Dropout(0.3),
        LSTM(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dropout(0.2),
        Dense(units=1)
    ])
    model_Close.compile(optimizer='adam', loss='mean_squared_error')
    model_Close.load_weights('best_model.h5')
    return model_Close

# Initialize global variables
ticker = "KC=F"
df = fetch_data_from_yahoo(ticker)
X, DataScaler_Close = normalize_data(df)

# Split data into samples
X_samples, y_samples = [], []
TimeSteps = 200  # next day's Price Prediction is based on last 200 past day's prices
for i in range(TimeSteps, len(X)):
    x_sample = X[i - TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)

X_data, y_data = np.array(X_samples), np.array(y_samples)

# Train on the last 300 days of data
X_train = X_data[-300:]
y_train = y_data[-300:]

# Define Input shapes for LSTM
TimeSteps = X_train.shape[1]
TotalFeatures = X_train.shape[2]

# Model initialization and training
model_Close = load_lstm_model(TimeSteps, TotalFeatures)

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    last_X_sample = np.array(req_data['last_X_sample']).reshape(1, TimeSteps, TotalFeatures)
    next_day_prediction = model_Close.predict(last_X_sample)
    next_day_price = DataScaler_Close.inverse_transform(next_day_prediction)[0][0]
    return jsonify({'predicted_price': next_day_price})

if __name__ == '__main__':
    app.run(debug=True)
