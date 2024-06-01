import argparse
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# Setup argument parser
parser = argparse.ArgumentParser(description='Run stock prediction model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
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
    tomorrow_prediction = model.predict(X_new)
    predicted_price = tomorrow_prediction[0]

    # Calculate the change from the last known price
    price_change = predicted_price - last_known_price
    price_change_percentage = (price_change / last_known_price) * 100

    # Determine the direction of the change
    direction = "UP" if price_change > 0 else "DOWN"

    # Print the prediction and the additional information
    print(f"Predicted Closing Price for Tomorrow: {predicted_price}")
    print(f"Expected change: {price_change:.2f} USD ({direction}), which is about {price_change_percentage:.2f}%")

# Main execution logic:
data = download_data('ABBV')
model, prepared_data = train_model(data)
predict_next_day(model, prepared_data)