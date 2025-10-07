import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Stock universe
STOCKS = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT', 'DIS', 'BA', 'PG', 'V', 'NVDA']

# Date range: 3 years from today
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

print(f"Downloading data from {start_date.date()} to {end_date.date()}")
print(f"Stocks: {', '.join(STOCKS)}\n")

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

all_data = {}

# Download data for each stock
for ticker in STOCKS:
    print(f"Downloading {ticker}...", end=" ")
    try:
        # Download using yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Clean the dataframe
        df.index.name = 'Date'
        
        # Save individual stock file
        filename = f"data/{ticker}_3y.csv"
        df.to_csv(filename)
        
        # Store adjusted close for combined file
        all_data[ticker] = df['Close']
        
        print(f"✅ {len(df)} rows")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*50)
print("Creating combined dataset...")

# Create combined dataframe
combined = pd.DataFrame(all_data)
combined.index.name = 'Date'
combined.to_csv('data/combined_3y.csv')

print(f"✅ Combined data saved to data/combined_3y.csv")
print(f"   Shape: {combined.shape[0]} rows x {combined.shape[1]} stocks")
print(f"   Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
print(f"\n   Stocks: {', '.join(combined.columns.tolist())}")

# Show summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS (3-Year Returns):\n")
returns = (combined.iloc[-1] / combined.iloc[0] - 1) * 100
returns = returns.sort_values(ascending=False)

for stock, ret in returns.items():
    print(f"   {stock:6s}: {ret:+7.2f}%")

print("\n✅ Data download complete!")
