# train.py
# Simple LSTM training example for closing price prediction.
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data(path, seq_len=60):
    df = pd.read_csv(path, parse_dates=['Date'])
    data = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_out', default='model.h5')
    args = parser.parse_args()

    X, y, scaler = load_data(args.data, seq_len=args.seq_len)
    model = build_model((X.shape[1], 1))
    checkpoint = ModelCheckpoint(args.model_out, save_best_only=True, monitor='loss', mode='min')
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, callbacks=[checkpoint])
    print(f"Trained and saved model to {args.model_out}")
