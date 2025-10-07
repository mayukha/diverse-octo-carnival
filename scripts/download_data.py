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

# Download data for each stock
for ticker in STOCKS:
    print(f"Downloading {ticker}...", end=" ")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Save to CSV
        filename = f"data/{ticker}_3y.csv"
        df.to_csv(filename)
        
        print(f"✅ Saved {len(df)} rows to {filename}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Data download complete!")

# Create a combined file for easy access
print("\nCreating combined dataset...")
all_data = {}
for ticker in STOCKS:
    df = pd.read_csv(f"data/{ticker}_3y.csv", index_col=0, parse_dates=True)
    all_data[ticker] = df['Adj Close']

combined = pd.DataFrame(all_data)
combined.to_csv('data/combined_3y.csv')
print(f"✅ Combined data saved to data/combined_3y.csv ({len(combined)} rows)")