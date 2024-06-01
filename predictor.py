import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# Get today's date in the format 'YYYY-MM-DD'
today_date = datetime.now().strftime('%Y-%m-%d')


# Step 1: Data Collection
abbv_data = yf.download('ABBV', start='2020-01-01', end=today_date)
print(abbv_data.head())  # To see the first few rows of the data
#abbv_data.to_csv('ABBV_stock_data.csv')

#plt.figure(figsize=(14, 7))
#plt.plot(abbv_data['Close'])
#plt.title('AbbVie Closing Stock Price')
#plt.xlabel('Date')
#plt.ylabel('Closing Price (USD)')
#plt.show()


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