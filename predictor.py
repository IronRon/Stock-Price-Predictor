import argparse
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Setup argument parser
parser = argparse.ArgumentParser(description='Run stock prediction model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
parser.add_argument('--compare', action='store_true', help='Compare today\'s prediction with actual closing price')
parser.add_argument('--maintain_csv', action='store_true', help='If specified, limits the CSV file to the last 30 entries to prevent it from growing too large.')
args = parser.parse_args()


def download_data(stock_symbol):
    # Get today's date in the format 'YYYY-MM-DD'
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Step 1: Data Collection
    return yf.download(stock_symbol, start='2020-01-01', end=today_date)

    #print(abbv_data.head())  # To see the first few rows of the data
    #abbv_data.to_csv('ABBV_stock_data.csv')

    #plt.figure(figsize=(14, 7))
    #plt.plot(abbv_data['Close'])
    #plt.title('AbbVie Closing Stock Price')
    #plt.xlabel('Date')
    #plt.ylabel('Closing Price (USD)')
    #plt.show()

def train_model(abbv_data):
    # Step 2: Data Preprocessing
    abbv_data['Previous_Close'] = abbv_data['Close'].shift(1)

    #Step 3: Feature Engineering
    abbv_data['7day_MA'] = abbv_data['Close'].rolling(window=7).mean()
    abbv_data['Daily_Change'] = abbv_data['Close'].pct_change()
    abbv_data = abbv_data[['Close', 'Previous_Close', '7day_MA', 'Daily_Change']]  # Keep only selected columns
    abbv_data = abbv_data.dropna() # Drop rows with any NaN values
    #print(abbv_data.isnull().sum())  # Print the count of NaNs in each column
    #print(abbv_data.head())

    #Step 4: Model Selection

    #Step 5: Data Splitting
    # Define features and target variable
    X = abbv_data[['Previous_Close', '7day_MA', 'Daily_Change']]
    y = abbv_data['Close']
    # Splitting the data into train and test sets

    if args.evaluate:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Step 6: Model Training
        # Creating and training the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        #Step 7: Making Predictions and Evaluating the Model
        # Making predictions
        predictions = model.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")


        #Step 8: Visualization
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line indicating perfect predictions
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Close Prices')
        plt.show()

        return model, abbv_data
    else:
        X_train, y_train = X[:-1], y[:-1]  # Use all data except the latest point for training

        #Step 6: Model Training
        # Creating and training the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, abbv_data

def predict_next_day(model, abbv_data):
    latest_data = abbv_data.iloc[-1]  # Get the most recent day's data
    last_known_price = latest_data['Close']

    # Prepare the input for prediction (reshape for a single sample)
    X_new = pd.DataFrame([[
        latest_data['Close'],  # Previous_Close as today's Close
        abbv_data['Close'].tail(7).mean(),  # 7day_MA from the last 7 days including today
        (latest_data['Close'] / abbv_data.iloc[-2]['Close'] - 1)  # Daily_Change as today's change
    ]], columns=['Previous_Close', '7day_MA', 'Daily_Change'])

    # Predict using the trained model
    predicted_price = model.predict(X_new)[0]
    # Save prediction with a date stamp for tomorrow
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Check if the prediction for the current day already exists
    try:
        with open("prediction_log.csv", "r") as file:
            existing_predictions = file.readlines()
    except FileNotFoundError:
        existing_predictions = []

    # Check if there's already an entry for tomorrow's date
    if not any(tomorrow_date in prediction for prediction in existing_predictions):
        with open("prediction_log.csv", "a") as file:
            file.write(f"{tomorrow_date},{predicted_price}\n")


    # Calculate the change from the last known price
    price_change = predicted_price - last_known_price
    price_change_percentage = (price_change / last_known_price) * 100

    # Determine the direction of the change
    direction = "UP" if price_change > 0 else "DOWN"

    # Print the prediction and the additional information
    print(f"Predicted Closing Price for Tomorrow ({tomorrow_date}): {predicted_price}")
    print(f"Expected change: {price_change:.2f} USD ({direction}), which is about {price_change_percentage:.2f}%")

def compare_prediction_with_actual(stock_symbol):

    # Fetch the predicted data
    try:
        with open("prediction_log.csv", "r") as file:
            predictions = file.readlines()
    except FileNotFoundError:
        print("No predictions found.")
        return

    for line in predictions:
        predicted_date, predicted_close = line.strip().split(',')
        predicted_close = float(predicted_close)

        # Set the end date to one day after the start date
        start_date = datetime.strptime(predicted_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1) # end date is exclusive

        # Format dates back to string for the yf.download function
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch actual data for the predicted date
        actual_data = yf.download(stock_symbol, start=start_date_str, end=end_date_str)
        if not actual_data.empty:
            actual_close = actual_data['Close'].iloc[-1]
            print(f"Actual Closing Price for {predicted_date}: {actual_close}")
            print(f"Predicted Closing Price was: {predicted_close}")

            difference = actual_close - predicted_close
            percentage_diff = (difference / predicted_close) * 100
            print(f"Difference: {difference:.2f} USD, which is about {percentage_diff:.2f}%")
        else:
            print(f"No trading data available for {predicted_date}.")


def maintain_csv_size(filepath, max_lines=30):
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Read the current content of the file
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Check if the file exceeds the maximum allowed lines
    if len(lines) > max_lines:
        # Keep only the last max_lines entries
        with open(filepath, 'w') as file:
            file.writelines(lines[-max_lines:])
        print(f"File exceeds the maximum allowed lines. Removed old entries...")
    else:
        print(f"File is chilling.")

# Main execution logic:
if args.compare:
    compare_prediction_with_actual('ABBV')
elif args.maintain_csv:
    # Call the function to maintain the size of the CSV file
    maintain_csv_size("prediction_log.csv", max_lines=30)
else:
    data = download_data('ABBV')
    model, prepared_data = train_model(data)
    predict_next_day(model, prepared_data)