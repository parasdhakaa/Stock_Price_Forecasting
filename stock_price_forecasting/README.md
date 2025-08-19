# Stock Price Forecasting

**What this project contains**
- `data_fetch.py` — script to download historical stock data using `yfinance`.
- `train.py` — training script (example LSTM) using TensorFlow/Keras.
- `predict.py` — load saved model and make predictions.
- `app.py` — Streamlit app for visualization and prediction.
- `requirements.txt` — Python dependencies.
- `sample_data.csv` — small sample dataset (generated).
- `.gitignore` — ignore venv and model files.

## How to use
1. Create a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate   # mac/linux
   .venv\Scripts\activate    # windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Download data for a ticker:
   ```
   python data_fetch.py --ticker AAPL --period 3y --output data.csv
   ```
4. Train model:
   ```
   python train.py --data data.csv --epochs 20 --model_out model.h5
   ```
5. Run Streamlit app:
   ```
   streamlit run app.py
   ```

## Notes
- The training script is a simple example using an LSTM. For improved results, try feature engineering,
  hyperparameter tuning, or alternative models (Prophet, Transformer-based, hybrid models).
- This repo is meant as a starting point — modify and extend it for a production-ready pipeline.
