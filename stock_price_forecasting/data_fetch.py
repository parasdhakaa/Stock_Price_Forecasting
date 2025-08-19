# data_fetch.py
# Fetch historical stock price data using yfinance and save to CSV.
import argparse
import yfinance as yf
import pandas as pd

def fetch(ticker, period='2y', interval='1d', output='data.csv'):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df.to_csv(output)
    print(f"Saved {len(df)} rows to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--period', default='2y')
    parser.add_argument('--interval', default='1d')
    parser.add_argument('--output', default='data.csv')
    args = parser.parse_args()
    fetch(args.ticker, period=args.period, interval=args.interval, output=args.output)
