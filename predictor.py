import argparse
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import io
import sys
import discord
from discord.ext import commands
import pandas_market_calendars as mcal
from pandas.tseries.holiday import AbstractHolidayCalendar
from pandas import date_range
from pandas import Timestamp

class AlwaysOpenCalendar(AbstractHolidayCalendar):
    rules = []

    def valid_days(self, start_date, end_date):
        # Returns every day between start_date and end_date inclusive
        return date_range(start=start_date, end=end_date, freq='D')

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Setup argument parser
#parser = argparse.ArgumentParser(description='Run stock prediction model.')
#parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
#parser.add_argument('--compare', action='store_true', help='Compare today\'s prediction with actual closing price')
#parser.add_argument('--maintain_csv', action='store_true', help='If specified, limits the CSV file to the last 30 entries to prevent it from growing too large.')
#args = parser.parse_args()


def download_data(stock_symbol):
    # Get today's date in the format 'YYYY-MM-DD'
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Step 1: Data Collection
    try:
        data = yf.download(stock_symbol, start='2020-01-01', end=today_date)
        return data
    except Exception as e:
        print(f"Failed to download data for {stock_symbol}: {e}")
        return None

    #print(abbv_data.head())  # To see the first few rows of the data
    #abbv_data.to_csv('ABBV_stock_data.csv')

    #plt.figure(figsize=(14, 7))
    #plt.plot(abbv_data['Close'])
    #plt.title('AbbVie Closing Stock Price')
    #plt.xlabel('Date')
    #plt.ylabel('Closing Price (USD)')
    #plt.show()

def train_model(abbv_data, evaluate):
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

    if evaluate:
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
        #print(f"Mean Squared Error: {mse}")
        #print(f"R^2 Score: {r2}")


        #Step 8: Visualization
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line indicating perfect predictions
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Close Prices')
        #plt.show()

        # Save the plot
        plt.savefig('plot.png')
        plt.close()  # Close the plot to free up memory

        return mse, r2  # Return the evaluation metrics
    else:
        X_train, y_train = X[:-1], y[:-1]  # Use all data except the latest point for training

        #Step 6: Model Training
        # Creating and training the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, abbv_data

def predict_next_day(model, abbv_data, stock_symbol):
    # Create a string buffer
    output = io.StringIO()
    sys.stdout = output  # Redirect stdout to the buffer

    # Get exchange and calendar for the stock
    exchange, asset_type = get_ticker_info(stock_symbol)
    calendar = get_calendar_by_exchange(exchange, asset_type)

    # Determine the next valid trading day
    today = datetime.now()
    next_trading_day = get_next_trading_day(today, calendar)

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
    #tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Reading and writing to the CSV with updated structure
    log_filename = "prediction_log.csv"
    if os.path.exists(log_filename):
        predictions = pd.read_csv(log_filename)
    else:
        predictions = pd.DataFrame(columns=["Symbol", "Date", "Predicted_Close"])

    # Ensure that predictions DataFrame has the necessary columns
    required_columns = ['Symbol', 'Date', 'Predicted_Close']
    for column in required_columns:
        if column not in predictions.columns:
            predictions[column] = pd.NA  # Initialize missing columns with NA

    if not ((predictions['Date'] == next_trading_day.strftime('%Y-%m-%d')) & (predictions['Symbol'] == stock_symbol)).any():
        # Create a new DataFrame for the row to be added
        new_row = pd.DataFrame({
            "Symbol": [stock_symbol],
            "Date": [next_trading_day.strftime('%Y-%m-%d')],
            "Predicted_Close": [predicted_price]
        })

        # Append new row using concat
        predictions = pd.concat([predictions, new_row], ignore_index=True)

        # Write to CSV, replacing or appending based on file existence
        if not os.path.exists("prediction_log.csv"):
            predictions.to_csv("prediction_log.csv", index=False)
        else:
            # Since we are appending to the CSV, we need to ensure only the new row is appended if file exists
            new_row.to_csv("prediction_log.csv", mode='a', header=False, index=False)

    # Calculate the change from the last known price
    price_change = predicted_price - last_known_price
    price_change_percentage = (price_change / last_known_price) * 100

    # Determine the direction of the change
    direction = "UP" if price_change > 0 else "DOWN"

    # Print the prediction and the additional information
    print(f"Predicted Closing Price for Next Trading Day ({next_trading_day.strftime('%Y-%m-%d')}): {predicted_price}")
    print(f"Expected change: {price_change:.2f} USD ({direction}), which is about {price_change_percentage:.2f}%")

    sys.stdout = sys.__stdout__  # Reset stdout
    return predicted_price, output.getvalue()

def compare_prediction_with_actual(stock_symbol):
    output = io.StringIO()
    sys.stdout = output  # Redirect stdout to the buffer

    # Fetch the predicted data
    try:
        predictions = pd.read_csv("prediction_log.csv")
        predictions = predictions[predictions['Symbol'] == stock_symbol]
    except FileNotFoundError:
        print("No predictions found.")
        sys.stdout = sys.__stdout__  # Reset stdout before returning
        return output.getvalue()

    for _, row in predictions.iterrows():
        predicted_date, predicted_close = row['Date'], row['Predicted_Close']

        # Set the end date to one day after the start date
        start_date = datetime.strptime(predicted_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1) # end date is exclusive

        # Format dates back to string for the yf.download function
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch actual data for the predicted date
        try:
            actual_data = yf.download(stock_symbol, start=start_date_str, end=end_date_str)
        except Exception as e:
            print(f"Failed to download data for {stock_symbol}: {e}")
            sys.stdout = sys.__stdout__  # Reset stdout before returning
            return output.getvalue()
        
        if not actual_data.empty:
            actual_close = actual_data['Close'].iloc[-1]
            print(f"Actual Closing Price for {predicted_date}: {actual_close}")
            print(f"Predicted Closing Price for {predicted_date} was: {predicted_close}")

            difference = actual_close - predicted_close
            percentage_diff = (difference / predicted_close) * 100
            print(f"Difference {predicted_date}: {difference:.2f} USD, which is about {percentage_diff:.2f}%")
        else:
            print(f"No trading data available for {predicted_date}.")

    sys.stdout = sys.__stdout__  # Reset stdout
    return output.getvalue()


def maintain_csv_size(filepath, max_entries=30):
    # Create a string buffer
    output = io.StringIO()
    sys.stdout = output  # Redirect stdout to the buffer

    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.stdout = sys.__stdout__  # Reset stdout
        return output.getvalue()
    
    # Read the current content of the file into a DataFrame
    df = pd.read_csv(filepath)

    # Process each symbol to limit entries
    result_df = pd.DataFrame()
    for symbol in df['Symbol'].unique():
        # Filter the dataframe by symbol and keep only the last max_entries entries
        symbol_df = df[df['Symbol'] == symbol]
        trimmed_df = symbol_df.tail(max_entries)
        result_df = pd.concat([result_df, trimmed_df], ignore_index=True)

    # Write the modified DataFrame back to the file
    result_df.to_csv(filepath, index=False)
    print(f"CSV file has been trimmed to the last {max_entries} entries for each stock.")
    
    sys.stdout = sys.__stdout__  # Reset stdout
    return output.getvalue()


def get_next_trading_day(current_date, calendar):
    # Make sure current_date is a Timestamp for consistency
    if not isinstance(current_date, Timestamp):
        current_date = Timestamp(current_date)

    valid_days = calendar.valid_days(start_date=current_date, end_date=current_date + timedelta(days=30))
    for day in valid_days:
        if day.date() > current_date.date():
            return day
    return None

        
def get_ticker_info(ticker):
    stock = yf.Ticker(ticker)
    # Fetch stock info
    info = stock.info
    # Get the 'exchange' attribute
    exchange = info.get('exchange')
    asset_type = info.get('quoteType', None)
    print(f"{ticker} is traded on: {exchange}")
    print(f"The asset {ticker} is a {asset_type}.")
    return exchange, asset_type


def get_calendar_by_exchange(exchange, asset_type):
    # Mapping of common exchange identifiers from yfinance to market calendars
    exchange_to_calendar = {
        'NMS': 'NASDAQ',   # Nasdaq
        'NYQ': 'NYSE',     # New York Stock Exchange
        'ASE': 'NYSE',     # NYSE American (formerly AMEX)
        'NYS': 'NYSE',     # NYSE
        'LSE': 'LSE',      # London Stock Exchange
        '24_7': AlwaysOpenCalendar(),  # For cryptocurrencies
        # add more mappings as necessary
    }

    if asset_type.lower() == 'cryptocurrency':
        return AlwaysOpenCalendar()
    else:
        calendar_name = exchange_to_calendar.get(exchange, '24_7')
        if isinstance(calendar_name, AbstractHolidayCalendar):
            return calendar_name
        elif calendar_name in mcal.get_calendar_names():
            return mcal.get_calendar(calendar_name)
        else:
            return AlwaysOpenCalendar()  # Default to 24/7 if no matching calendar


@bot.command()
async def predict(ctx, symbol: str):
    """Predicts the next day closing price for a given stock symbol."""
    data = download_data(symbol)
    model, prepared_data = train_model(data, False)
    predicted_price, analysis  = predict_next_day(model, prepared_data, symbol)
    embed = discord.Embed(title=f"Prediction for {symbol}", color=0x00ff00)
    embed.add_field(name="Predicted Closing Price", value=f"${predicted_price:.2f}", inline=False)
    embed.add_field(name="Details", value=analysis, inline=False)
    await ctx.send(embed=embed)

@bot.command()
async def compare(ctx, symbol: str):
    """Compares the predicted with the actual closing price."""
    comparison_results = compare_prediction_with_actual(symbol)
    if comparison_results:
        # Split the results into lines for better formatting within fields
        results_lines = comparison_results.split('\n')
        
        # Create an embed object for response
        embed = discord.Embed(title=f"Comparison Results for {symbol}", description="Here are the price comparisons between predicted and actual closing prices:", color=0x3498db)
        
        # Add fields dynamically based on content length
        for line in results_lines:
            if line:  # Avoid adding empty lines
                # Safely split the line and handle lines without a colon
                parts = line.split(':')
                if len(parts) == 2:
                    field_name, field_value = parts
                    embed.add_field(name=field_name.strip(), value=field_value.strip(), inline=False)
                else:
                    # Handle lines without a colon by adding them as a generic field
                    embed.add_field(name="Notice", value=line, inline=False)
        
        # Send the embed
        await ctx.send(embed=embed)
    else:
        await ctx.send("No comparison results to display.")

@bot.command()
async def maintain(ctx, max_entries=10):
    """Maintain the size of the prediction log CSV file."""
    result = maintain_csv_size("prediction_log.csv", max_entries)
    embed = discord.Embed(title="CSV Maintenance Report", description="Maintenance operations completed on prediction log CSV file.", color=0x3498db)
    embed.add_field(name="Result", value=result, inline=False)
    embed.set_footer(text="Maintenance executed successfully.")

    await ctx.send(embed=embed)

@bot.command()
async def evaluate(ctx, symbol):
    data = download_data(symbol)
    mse, r2 = train_model(data, True)  # Capture the returned values

    # Create an embed object for the response
    embed = discord.Embed(
        title="Model Evaluation Results",
        description=f"Results for the symbol: **{symbol.upper()}**",
        color=discord.Color.blue()  # You can choose any appropriate color
    )
    embed.add_field(name="Mean Squared Error (MSE)", value=f"`{mse:.4f}`", inline=False)
    embed.add_field(name="R² Score", value=f"`{r2:.4f}`", inline=False)
    embed.set_footer(text="Model evaluation completed")

    # Send the plot and the embed message to the Discord channel
    with open('plot.png', 'rb') as f:
        picture = discord.File(f)
        await ctx.send(file=picture, embed=embed)

@bot.command()
async def info(ctx):
    """Provides information about the bot and usage instructions."""
    # Create an embed object for the response
    embed = discord.Embed(
        title="Bot Information and Usage Guide",
        description="This bot provides advanced stock market analysis features. Below are the commands you can use:",
        color=discord.Color.gold()  # Setting a gold color for the embed
    )

    # Adding fields for each command with a brief description
    embed.add_field(
        name="!predict [symbol]",
        value="Predicts the next day's closing price for the specified stock symbol. Example: `@predict AAPL`",
        inline=False
    )
    embed.add_field(
        name="!compare [symbol]",
        value="Compares the predicted and actual closing prices for the specified stock symbol. Example: `@compare AAPL`",
        inline=False
    )
    embed.add_field(
        name="!maintain [max_entries]",
        value="Maintains the size of the prediction log to ensure it doesn't exceed the specified maximum entries. Default is 10. Example: `@maintain 15`",
        inline=False
    )
    embed.add_field(
        name="!evaluate [symbol]",
        value="Evaluates the prediction model using the specified stock symbol, providing model accuracy metrics like MSE and R² Score. Example: `@evaluate AAPL`",
        inline=False
    )

    # Adding a footer for additional context or help
    embed.set_footer(text="For more details, visit https://github.com/IronRon/Stock-Price-Predictor/tree/discord-bot-feature")

    # Send the embed message to the Discord channel
    await ctx.send(embed=embed)

# Main execution logic:
#if args.compare:
#    compare_prediction_with_actual('ABBV')
#elif args.maintain_csv:
#    # Call the function to maintain the size of the CSV file
#    maintain_csv_size("prediction_log.csv", max_lines=30)
#else:
#    data = download_data('ABBV')
#    model, prepared_data = train_model(data, args.evaluate)
#    predict_next_day(model, prepared_data)

bot.run('YOUR-BOT-CODE')