# Stock Price Prediction and Evaluation Discord Bot

This project employs machine learning techniques, specifically linear regression, to predict the closing stock prices of various companies. It utilizes historical stock data from Yahoo Finance for model training and predictions. The project is designed to be used with a Discord bot, allowing users to interact with the prediction system through Discord commands.

## Features

1. **Data Collection**: Retrieves historical stock data for AbbVie from Yahoo Finance starting from January 1, 2020, to the current date.
2. **Data Preprocessing**: Cleans and prepares the data for modeling, including handling missing values.
3. **Feature Engineering**: Enhances the dataset with additional features such as the previous day's close, a 7-day moving average, and daily percentage change.
4. **Model Training**: Trains a linear regression model using the processed data.
5. **Prediction**: Predicts the next day's closing price using the most recent data available.
6. **Model Evaluation** (Optional): Evaluates the model's accuracy using metrics like mean squared error (MSE) and R² score. Provides visualization for actual vs. predicted prices.
7. **Comparison with Actual Prices**: Compares model predictions with actual closing prices on subsequent days.
8. **CSV Maintenance**: Manages the size of log files to prevent them from growing too large by limiting them to the last 30 entries.

## Requirements

- **Python**: Version 3.x
- **Libraries**: 
  - `yfinance`: For downloading financial data.
  - `pandas`: For data manipulation and analysis.
  - `scikit-learn`: For implementing machine learning models.
  - `matplotlib`: For plotting graphs.
  - `argparse`: For parsing command-line options and arguments.
  - `discord.py`: For creating and handling the Discord bot.

## Usage
The project is designed to be run within a Discord bot environment. Below are the commands that users can issue to interact with the bot:
- **Predict Mode:** This mode will run the script to fetch the latest stock data, train the model, and predict the next day’s closing price without evaluating the model. 
  ```
  !predict <symbol>
  ```
- **Evaluation Mode:** This mode includes all the steps in the normal mode and also evaluates the model's performance and visualizes the results. 
  ```
  !evaluate <symbol>
  ```
- **Comparison Mode:** Compares the predicted prices with actual closing prices on the corresponding dates. 
  ```
  !compare <symbol>
  ```
- **CSV Maintenance Mode:** Manages the size of the prediction log file by ensuring it does not exceed 30 entries.
  ```
  !maintain
  ```
Each of these commands will interact with the bot and the underlying machine learning model to provide users with predictions, comparisons, and evaluations. The bot will respond with rich embed messages that display the requested data and analyses.