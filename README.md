# Stock Price Prediction and Evaluation

This project involves a Python script that utilizes machine learning techniques to predict the closing stock prices for AbbVie (ticker symbol: ABBV) and optionally evaluates the performance of the model. The script fetches historical stock data, processes it, trains a linear regression model, and predicts the next day's closing price. It also allows users to optionally evaluate the model's accuracy through mean squared error and R² score, and visualize the prediction results.

## Features

- **Data Collection:** Fetches historical stock data for AbbVie from Yahoo Finance.
- **Data Preprocessing:** Processes the data to prepare it for the machine learning model.
- **Feature Engineering:** Enhances the dataset with additional features like previous close, 7-day moving average, and daily percentage change.
- **Model Training:** Trains a linear regression model on the processed data.
- **Prediction:** Predicts the next day's closing price based on the latest available data.
- **Optional Evaluation:** Optionally evaluates the model's accuracy and visualizes the actual vs. predicted prices.

## Requirements

- Python 3.x
- Libraries: yfinance, pandas, scikit-learn, matplotlib

## Usage
The script can be run in two modes: normal and evaluation.
- **Normal Mode:** This mode will run the script to fetch the latest stock data, train the model, and predict the next day’s closing price without evaluating the model. python stock_prediction.py
- **Evaluation Mode:** This mode includes all the steps in the normal mode and also evaluates the model's performance and visualizes the results. python stock_prediction.py --evaluate