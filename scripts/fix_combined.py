import pandas as pd
import os

STOCKS = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT', 'DIS', 'BA', 'PG', 'V', 'NVDA']

print("Checking data files...\n")

# First, let's see what columns we actually have
sample = pd.read_csv('data/AAPL_3y.csv', nrows=5)
print("Sample columns from AAPL file:")
print(sample.columns.tolist())
print("\nFirst few rows:")
print(sample.head())

# Now create combined dataset
all_data = {}
for ticker in STOCKS:
    print(f"\nProcessing {ticker}...", end=" ")
    df = pd.read_csv(f"data/{ticker}_3y.csv", index_col=0)
    
    # Check which price column exists
    if 'Adj Close' in df.columns:
        all_data[ticker] = df['Adj Close']
    elif 'Close' in df.columns:
        all_data[ticker] = df['Close']
    else:
        print(f"Available columns: {df.columns.tolist()}")
        continue
    
    print(f"✅ {len(df)} rows")

combined = pd.DataFrame(all_data)
combined.index = pd.to_datetime(combined.index)
combined.to_csv('data/combined_3y.csv')
print(f"\n✅ Combined data saved: {len(combined)} rows x {len(combined.columns)} stocks")
print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
