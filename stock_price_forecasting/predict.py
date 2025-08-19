# predict.py
# Load model and predict next day's closing price given latest seq_len days.
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict(model_path, data_path, seq_len=60):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    data = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    last_seq = data_scaled[-seq_len:]
    X = np.array([last_seq])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = load_model(model_path)
    pred_scaled = model.predict(X)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    print(f"Predicted next close: {pred:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.h5')
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--seq_len', type=int, default=60)
    args = parser.parse_args()
    predict(args.model, args.data, args.seq_len)
