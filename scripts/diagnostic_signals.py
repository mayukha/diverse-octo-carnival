import pandas as pd
import numpy as np

# Load data and calculate indicators (same as backtest)
stocks = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT', 'DIS', 'BA', 'PG', 'V', 'NVDA']
dfs = {}

for stock in stocks:
    filepath = f'data/{stock}_3y.csv'
    stock_df = pd.read_csv(filepath, parse_dates=['Date'])
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    
    # Calculate indicators
    stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
    stock_df['Volume_MA50'] = stock_df['Volume'].rolling(window=50).mean()
    stock_df['Volume_Ratio'] = stock_df['Volume'] / stock_df['Volume_MA50']
    stock_df['Peak'] = stock_df['Close'].expanding().max()
    stock_df['Drawdown'] = (stock_df['Close'] - stock_df['Peak']) / stock_df['Peak']
    stock_df['MA_Above'] = (stock_df['MA20'] > stock_df['MA50']).astype(int)
    stock_df['MA_Cross'] = stock_df['MA_Above'].diff()
    
    dfs[stock] = stock_df

print("="*80)
print("DIAGNOSTIC: WHY NO TRADES?")
print("="*80)

# Check each filter
for stock in stocks:
    df = dfs[stock]
    
    # Find MA crossovers
    bullish_crosses = df[df['MA_Cross'] == 1].copy()
    
    if len(bullish_crosses) > 0:
        print(f"\n{stock}: Found {len(bullish_crosses)} bullish MA crosses")
        
        # Check each crossover against filters
        for idx, row in bullish_crosses.iterrows():
            date = row['Date']
            
            # Check filters
            volume_ok = row['Volume_Ratio'] >= 1.5
            drawdown_ok = row['Drawdown'] >= -0.15
            
            status = "✅ WOULD ENTER" if (volume_ok and drawdown_ok) else "❌ BLOCKED"
            
            print(f"  {date.strftime('%Y-%m-%d')}: {status}")
            if not volume_ok:
                print(f"    ❌ Volume: {row['Volume_Ratio']:.2f}x (need 1.5x)")
            else:
                print(f"    ✅ Volume: {row['Volume_Ratio']:.2f}x")
            
            if not drawdown_ok:
                print(f"    ❌ Drawdown: {row['Drawdown']*100:.1f}% (need > -15%)")
            else:
                print(f"    ✅ Drawdown: {row['Drawdown']*100:.1f}%")
    else:
        print(f"\n{stock}: ❌ NO bullish MA crosses found")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_crosses = sum(len(dfs[s][dfs[s]['MA_Cross'] == 1]) for s in stocks)
print(f"Total bullish crosses across all stocks: {total_crosses}")

# Count how many would pass each filter
volume_pass = 0
drawdown_pass = 0
both_pass = 0

for stock in stocks:
    df = dfs[stock]
    crosses = df[df['MA_Cross'] == 1].copy()
    
    for idx, row in crosses.iterrows():
        volume_ok = row['Volume_Ratio'] >= 1.5
        drawdown_ok = row['Drawdown'] >= -0.15
        
        if volume_ok:
            volume_pass += 1
        if drawdown_ok:
            drawdown_pass += 1
        if volume_ok and drawdown_ok:
            both_pass += 1

print(f"\nFilter Pass Rates:")
print(f"  Volume filter (1.5x):        {volume_pass}/{total_crosses} ({volume_pass/total_crosses*100:.1f}%)")
print(f"  Drawdown filter (-15%):      {drawdown_pass}/{total_crosses} ({drawdown_pass/total_crosses*100:.1f}%)")
print(f"  Both filters:                {both_pass}/{total_crosses} ({both_pass/total_crosses*100:.1f}%)")

print("\n💡 RECOMMENDATIONS:")
if both_pass == 0:
    print("  - Filters are TOO STRICT - not a single signal passes!")
    print("  - Try: Remove drawdown filter OR lower volume to 1.2x")
elif both_pass < 10:
    print("  - Very few signals passing - filters may be too conservative")
    print("  - Consider loosening one filter slightly")
else:
    print("  - Signals are being generated - issue must be elsewhere")
    print("  - Check: Correlation filter or 'wait for 3 signals' rule")
    