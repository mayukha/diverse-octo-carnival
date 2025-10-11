import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================
MA_SHORT = 20
MA_LONG = 50
VOLUME_THRESHOLD = 1.2  # Must be 1.2x the 50-day average
MAX_DRAWDOWN_ENTRY = -0.15  # Don't enter if stock is down >15% from peak
STOP_LOSS = -0.03  # Exit at -3%
TAKE_PROFIT = 0.50  # Exit at +50%
MAX_POSITIONS = 5
MIN_POSITIONS_TO_START = 1
CORRELATION_THRESHOLD = 0.75  # Max correlation with existing positions
CORRELATION_WINDOW = 50  # Days to calculate rolling correlation

INITIAL_CAPITAL = 100000

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

# Load individual stock files to get OHLCV data
stocks = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT', 'DIS', 'BA', 'PG', 'V', 'NVDA']
dfs = {}

for stock in stocks:
    filepath = f'data/{stock}_3y.csv'
    stock_df = pd.read_csv(filepath, parse_dates=['Date'])
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    stock_df['Ticker'] = stock
    dfs[stock] = stock_df

print(f"Loaded {len(stocks)} stocks")
print(f"Date range: {dfs['AAPL']['Date'].min()} to {dfs['AAPL']['Date'].max()}")
print(f"Total trading days: {len(dfs['AAPL'])}\n")

# ============================================================================
# CALCULATE INDICATORS FOR ALL STOCKS
# ============================================================================
print("Calculating indicators...")

def calculate_indicators(stock_df):
    """Calculate all technical indicators for a stock"""
    stock_df = stock_df.copy()
    
    # Moving averages
    stock_df['MA20'] = stock_df['Close'].rolling(window=MA_SHORT).mean()
    stock_df['MA50'] = stock_df['Close'].rolling(window=MA_LONG).mean()
    
    # Volume average
    stock_df['Volume_MA50'] = stock_df['Volume'].rolling(window=MA_LONG).mean()
    stock_df['Volume_Ratio'] = stock_df['Volume'] / stock_df['Volume_MA50']
    
    # Drawdown from peak
    stock_df['Peak'] = stock_df['Close'].expanding().max()
    stock_df['Drawdown'] = (stock_df['Close'] - stock_df['Peak']) / stock_df['Peak']
    
    # MA crossover signals (1 = bullish cross, -1 = bearish cross, 0 = no cross)
    stock_df['MA_Above'] = (stock_df['MA20'] > stock_df['MA50']).astype(int)
    stock_df['MA_Cross'] = stock_df['MA_Above'].diff()
    
    # Daily returns for correlation calculation
    stock_df['Returns'] = stock_df['Close'].pct_change()
    
    return stock_df

# Apply indicators to each stock
for stock in stocks:
    dfs[stock] = calculate_indicators(dfs[stock])

# ============================================================================
# BACKTEST ENGINE
# ============================================================================
print("\nRunning backtest...")

class Position:
    def __init__(self, ticker, entry_date, entry_price, shares, capital_allocated):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.capital_allocated = capital_allocated
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.return_pct = None
        
    def update_value(self, current_price):
        return self.shares * current_price
    
    def close(self, exit_date, exit_price, reason):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.return_pct = (exit_price - self.entry_price) / self.entry_price

# Portfolio state
cash = INITIAL_CAPITAL
positions = {}  # {ticker: Position}
closed_trades = []
portfolio_values = []
dates = []

# Get all unique dates (from any stock, they should all be the same)
all_dates = sorted(dfs['AAPL']['Date'].unique())

# Helper function to get stock data for a specific date
def get_stock_data(ticker, date):
    stock_df = dfs[ticker]
    data = stock_df[stock_df['Date'] == date]
    return data.iloc[0] if not data.empty else None

# Helper function to calculate correlation matrix for current date
def get_correlation_matrix(date, window=CORRELATION_WINDOW):
    """Calculate rolling correlation of returns for all stocks"""
    corr_data = {}
    
    for stock in stocks:
        stock_df = dfs[stock]
        stock_df_filtered = stock_df[stock_df['Date'] <= date].tail(window)
        if len(stock_df_filtered) >= window:
            corr_data[stock] = stock_df_filtered['Returns'].values
    
    if len(corr_data) < 2:
        return None
    
    corr_df = pd.DataFrame(corr_data)
    return corr_df.corr()

# Helper function to check correlation with existing positions
def check_correlation(ticker, date, existing_tickers):
    """Check if ticker has correlation < threshold with all existing positions"""
    if not existing_tickers:
        return True
    
    corr_matrix = get_correlation_matrix(date)
    if corr_matrix is None:
        return True  # Not enough data, allow entry
    
    for existing in existing_tickers:
        if existing in corr_matrix.index and ticker in corr_matrix.index:
            corr_value = corr_matrix.loc[ticker, existing]
            if abs(corr_value) >= CORRELATION_THRESHOLD:
                return False
    
    return True

# Main backtest loop
for date in all_dates:
    dates.append(date)
    
    # Get current data for all stocks
    current_data = {}
    for stock in stocks:
        data = get_stock_data(stock, date)
        if data is not None:
            current_data[stock] = data
    
    # ========================================================================
    # 1. CHECK EXITS (do this first before entries)
    # ========================================================================
    positions_to_close = []
    
    for ticker, pos in positions.items():
        if ticker not in current_data:
            continue
            
        data = current_data[ticker]
        current_price = data['Close']
        
        # Calculate current return
        current_return = (current_price - pos.entry_price) / pos.entry_price
        
        # Exit conditions
        exit_reason = None
        
        # 1. MA bearish crossover
        if data['MA_Cross'] == -1:
            exit_reason = 'MA_Cross'
        
        # 2. Stop loss
        elif current_return <= STOP_LOSS:
            exit_reason = 'Stop_Loss'
        
        # 3. Take profit
        elif current_return >= TAKE_PROFIT:
            exit_reason = 'Take_Profit'
        
        if exit_reason:
            pos.close(date, current_price, exit_reason)
            cash += pos.update_value(current_price)
            closed_trades.append(pos)
            positions_to_close.append(ticker)
    
    # Remove closed positions
    for ticker in positions_to_close:
        del positions[ticker]
    
    # Rebalance remaining positions if any closed
    if positions_to_close and positions:
        # Redistribute capital equally among remaining positions
        total_equity = cash + sum(pos.update_value(current_data[t]['Close']) 
                                 for t, pos in positions.items() if t in current_data)
        position_size = total_equity / len(positions)
        
        for ticker, pos in positions.items():
            if ticker in current_data:
                current_price = current_data[ticker]['Close']
                pos.shares = position_size / current_price
                pos.capital_allocated = position_size
        
        cash = total_equity - sum(pos.capital_allocated for pos in positions.values())
    
    # ========================================================================
    # 2. CHECK ENTRIES (only if we have room for more positions)
    # ========================================================================
    if len(positions) < MAX_POSITIONS:
        entry_candidates = []
        
        for stock in stocks:
            # Skip if we already hold this stock
            if stock in positions:
                continue
            
            if stock not in current_data:
                continue
            
            data = current_data[stock]
            
            # Need enough data for indicators
            if pd.isna(data['MA20']) or pd.isna(data['MA50']) or pd.isna(data['Volume_Ratio']):
                continue
            
            # Check all entry conditions
            # 1. MA bullish crossover
            if data['MA_Cross'] != 1:
                continue
            
            # 2. Volume confirmation
            if data['Volume_Ratio'] < VOLUME_THRESHOLD:
                continue
            
            # 3. Drawdown filter
            if data['Drawdown'] < MAX_DRAWDOWN_ENTRY:
                continue
            
            # 4. Correlation check
            existing_tickers = list(positions.keys())
            if not check_correlation(stock, date, existing_tickers):
                continue
            
            # All conditions met - this is a candidate
            entry_candidates.append(stock)
        
        # Sort candidates by correlation (lowest correlation to portfolio first)
        if entry_candidates:
            corr_matrix = get_correlation_matrix(date)
            
            if corr_matrix is not None and positions:
                # Calculate average correlation with existing positions
                avg_corrs = []
                for candidate in entry_candidates:
                    corrs = []
                    for existing in positions.keys():
                        if existing in corr_matrix.index and candidate in corr_matrix.index:
                            corrs.append(abs(corr_matrix.loc[candidate, existing]))
                    avg_corr = np.mean(corrs) if corrs else 0
                    avg_corrs.append((candidate, avg_corr))
                
                # Sort by lowest average correlation
                avg_corrs.sort(key=lambda x: x[1])
                entry_candidates = [x[0] for x in avg_corrs]
        
        # Enter positions
        for candidate in entry_candidates:
            if len(positions) >= MAX_POSITIONS:
                break
            
            # Check if we have enough positions to start investing
            total_positions = len(positions) + 1
            
            # If we're below minimum, accumulate cash
            if total_positions < MIN_POSITIONS_TO_START:
                continue
            
            # Calculate capital allocation
            total_equity = cash + sum(pos.update_value(current_data[t]['Close']) 
                                     for t, pos in positions.items() if t in current_data)
            
            # Rebalance: equal weight across all positions including new one
            target_positions = len(positions) + 1
            position_size = total_equity / target_positions
            
            # Adjust existing positions
            for ticker, pos in positions.items():
                if ticker in current_data:
                    current_price = current_data[ticker]['Close']
                    old_value = pos.update_value(current_price)
                    
                    if old_value > position_size:
                        # Reduce position, free up cash
                        reduction = old_value - position_size
                        pos.shares = position_size / current_price
                        pos.capital_allocated = position_size
                        cash += reduction
                    else:
                        # Increase position if we have cash
                        addition = position_size - old_value
                        if cash >= addition:
                            pos.shares = position_size / current_price
                            pos.capital_allocated = position_size
                            cash -= addition
            
            # Enter new position if we have enough cash
            if cash >= position_size:
                entry_price = current_data[candidate]['Close']
                shares = position_size / entry_price
                
                positions[candidate] = Position(
                    ticker=candidate,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    capital_allocated=position_size
                )
                
                cash -= position_size
    
    # ========================================================================
    # 3. CALCULATE PORTFOLIO VALUE
    # ========================================================================
    position_value = sum(pos.update_value(current_data[ticker]['Close']) 
                        for ticker, pos in positions.items() 
                        if ticker in current_data)
    
    total_value = cash + position_value
    portfolio_values.append(total_value)

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================
print("\nBacktest complete!")
print(f"Total trades: {len(closed_trades)}")
print(f"Final positions open: {len(positions)}")

# Close any remaining open positions at final price
for ticker, pos in positions.items():
    final_data = dfs[ticker].iloc[-1]
    pos.close(final_data['Date'], final_data['Close'], 'End_of_Period')
    closed_trades.append(pos)

# Create performance dataframe
perf_df = pd.DataFrame({
    'Date': dates,
    'Portfolio_Value': portfolio_values
})

perf_df['Returns'] = perf_df['Portfolio_Value'].pct_change()
perf_df['Cumulative_Return'] = (perf_df['Portfolio_Value'] / INITIAL_CAPITAL - 1) * 100

# Calculate metrics
final_value = portfolio_values[-1]
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
winning_trades = [t for t in closed_trades if t.return_pct > 0]
losing_trades = [t for t in closed_trades if t.return_pct <= 0]

win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
avg_win = np.mean([t.return_pct for t in winning_trades]) * 100 if winning_trades else 0
avg_loss = np.mean([t.return_pct for t in losing_trades]) * 100 if losing_trades else 0

# Sharpe ratio (annualized)
returns = perf_df['Returns'].dropna()
sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0

# Max drawdown
running_max = perf_df['Portfolio_Value'].expanding().max()
drawdown = (perf_df['Portfolio_Value'] - running_max) / running_max
max_drawdown = drawdown.min() * 100

# ============================================================================
# BENCHMARK: EQUAL WEIGHT BUY & HOLD
# ============================================================================
benchmark_values = []
benchmark_allocation = INITIAL_CAPITAL / len(stocks)

for date in all_dates:
    total = 0
    for stock in stocks:
        data = get_stock_data(stock, date)
        if data is not None:
            price = data['Close']
            initial_price = dfs[stock].iloc[0]['Close']
            shares = benchmark_allocation / initial_price
            total += shares * price
    benchmark_values.append(total)

benchmark_return = (benchmark_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# ============================================================================
# PRINT RESULTS
# ============================================================================
print("\n" + "="*60)
print("STRATEGY PERFORMANCE")
print("="*60)
print(f"Initial Capital:        ${INITIAL_CAPITAL:,.0f}")
print(f"Final Value:            ${final_value:,.0f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Benchmark (B&H):        {benchmark_return:.2f}%")
print(f"Alpha:                  {total_return - benchmark_return:.2f}%")
print(f"\nRisk Metrics:")
print(f"Max Drawdown:           {max_drawdown:.2f}%")
print(f"Sharpe Ratio:           {sharpe:.3f}")
print(f"\nTrade Statistics:")
print(f"Total Trades:           {len(closed_trades)}")
print(f"Winning Trades:         {len(winning_trades)} ({win_rate:.1f}%)")
print(f"Losing Trades:          {len(losing_trades)}")
print(f"Average Win:            {avg_win:.2f}%")
print(f"Average Loss:           {avg_loss:.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_dir = 'results/moving_average_optimized'
os.makedirs(output_dir, exist_ok=True)

# Save trade log
trade_log = pd.DataFrame([{
    'Ticker': t.ticker,
    'Entry_Date': t.entry_date,
    'Entry_Price': t.entry_price,
    'Exit_Date': t.exit_date,
    'Exit_Price': t.exit_price,
    'Return_%': t.return_pct * 100,
    'Exit_Reason': t.exit_reason,
    'Capital_Allocated': t.capital_allocated
} for t in closed_trades])

trade_log.to_csv(f'{output_dir}/trade_log.csv', index=False)
print(f"\nTrade log saved to: {output_dir}/trade_log.csv")

# Save performance data
perf_df.to_csv(f'{output_dir}/portfolio_performance.csv', index=False)
print(f"Performance data saved to: {output_dir}/portfolio_performance.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Moving Average Strategy - Performance Analysis', fontsize=16, fontweight='bold')

# 1. Equity Curve
ax1 = axes[0, 0]
ax1.plot(perf_df['Date'], perf_df['Portfolio_Value'], label='Strategy', linewidth=2)
ax1.plot(perf_df['Date'], benchmark_values, label='Buy & Hold (Equal Weight)', 
         linewidth=2, alpha=0.7, linestyle='--')
ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
ax1.set_title('Portfolio Value Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 2. Cumulative Returns
ax2 = axes[0, 1]
benchmark_returns = [(v - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for v in benchmark_values]
ax2.plot(perf_df['Date'], perf_df['Cumulative_Return'], label='Strategy', linewidth=2)
ax2.plot(perf_df['Date'], benchmark_returns, label='Buy & Hold', 
         linewidth=2, alpha=0.7, linestyle='--')
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_title('Cumulative Returns')
ax2.set_xlabel('Date')
ax2.set_ylabel('Return (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Drawdown
ax3 = axes[1, 0]
ax3.fill_between(perf_df['Date'], drawdown * 100, 0, alpha=0.3, color='red')
ax3.plot(perf_df['Date'], drawdown * 100, color='red', linewidth=1)
ax3.set_title('Portfolio Drawdown')
ax3.set_xlabel('Date')
ax3.set_ylabel('Drawdown (%)')
ax3.grid(True, alpha=0.3)

# 4. Trade Distribution
ax4 = axes[1, 1]
returns_pct = [t.return_pct * 100 for t in closed_trades]
ax4.hist(returns_pct, bins=30, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.axvline(x=np.mean(returns_pct), color='green', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(returns_pct):.1f}%')
ax4.set_title('Trade Return Distribution')
ax4.set_xlabel('Return (%)')
ax4.set_ylabel('Number of Trades')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
print(f"Performance charts saved to: {output_dir}/performance_summary.png")

print("\n✅ Backtest complete!")