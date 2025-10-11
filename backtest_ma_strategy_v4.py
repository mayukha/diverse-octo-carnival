import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================
# V4.0 CONFIGURATION - SMART EXITS
# ============================
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
           'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'BAC',
           'XOM', 'CVX', 'ABBV', 'PFE', 'COST', 'DIS', 'NFLX',
           'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'BA', 'NKE']

START_DATE = '2022-10-01'
END_DATE = '2025-10-11'
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.12  # 12% per position
MAX_POSITIONS = 7

# Entry Rules (SAME AS V3)
FAST_MA = 20
SLOW_MA = 50
VOLUME_THRESHOLD = 1.1

# Exit Rules (NEW!)
MIN_HOLD_DAYS = 90  # Prevent whipsaws
MAX_HOLD_DAYS = 180  # Force turnover
TAKE_PROFIT = 0.50  # +50%
TRAILING_STOP = 0.20  # 20% from peak

print("=" * 80)
print("MA STRATEGY V4.0 - SMART EXITS")
print("=" * 80)
print(f"\n🎯 THE FIX: Keep good entries, fix terrible exits!")
print(f"\nEntry: {FAST_MA}/{SLOW_MA} MA crossover + {VOLUME_THRESHOLD}x volume")
print(f"\nExit Rules (NEW):")
print(f"  1. Minimum hold: {MIN_HOLD_DAYS} days (no whipsaws)")
print(f"  2. Trailing stop: {TRAILING_STOP*100}% from peak (let winners run)")
print(f"  3. Take profit: +{TAKE_PROFIT*100}% (keep WMT-style wins)")
print(f"  4. Maximum hold: {MAX_HOLD_DAYS} days (force rotation)")
print(f"  5. MA crossunder: ONLY after {MIN_HOLD_DAYS} days")
print(f"\nPortfolio: Max {MAX_POSITIONS} positions @ {POSITION_SIZE*100}% each")
print("=" * 80)

# ============================
# DATA DOWNLOAD
# ============================
print("\n📊 Downloading data...")
data = {}
for ticker in TICKERS:
    try:
        # Download with multi-index handling
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) > SLOW_MA * 2:
            data[ticker] = df
            print(f"  ✓ {ticker}: {len(df)} days")
    except Exception as e:
        print(f"  ✗ {ticker}: Error - {e}")

print(f"\n✓ Loaded {len(data)} tickers")

# ============================
# TECHNICAL INDICATORS
# ============================
print("\n📈 Calculating indicators...")
for ticker in data:
    df = data[ticker]
    
    # Moving averages
    df['MA_fast'] = df['Close'].rolling(window=FAST_MA).mean()
    df['MA_slow'] = df['Close'].rolling(window=SLOW_MA).mean()
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=SLOW_MA).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Momentum for priority
    df['Momentum'] = df['Close'].pct_change(20)
    
    # Crossover detection
    df['MA_cross'] = 0
    df.loc[(df['MA_fast'] > df['MA_slow']) & 
           (df['MA_fast'].shift(1) <= df['MA_slow'].shift(1)), 'MA_cross'] = 1
    
    # Crossunder detection
    df['MA_crossunder'] = 0
    df.loc[(df['MA_fast'] < df['MA_slow']) & 
           (df['MA_fast'].shift(1) >= df['MA_slow'].shift(1)), 'MA_crossunder'] = 1

print("✓ Indicators calculated")

# ============================
# BACKTESTING ENGINE
# ============================
class Position:
    def __init__(self, ticker, entry_date, entry_price, shares):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.peak_price = entry_price  # Track peak for trailing stop
        self.hold_days = 0
    
    def update_peak(self, current_price):
        """Update peak price for trailing stop"""
        self.peak_price = max(self.peak_price, current_price)
    
    def check_exit(self, current_date, current_price, ma_crossunder):
        """Check all exit conditions"""
        self.hold_days = (current_date - self.entry_date).days
        return_pct = (current_price - self.entry_price) / self.entry_price
        
        # Update peak
        self.update_peak(current_price)
        
        # Exit condition 1: Take profit (+50%)
        if return_pct >= TAKE_PROFIT:
            return True, 'take_profit', return_pct
        
        # Exit condition 2: Max hold period (180 days)
        if self.hold_days >= MAX_HOLD_DAYS:
            return True, 'max_hold', return_pct
        
        # Exit condition 3: Trailing stop (20% from peak)
        drawdown_from_peak = (self.peak_price - current_price) / self.peak_price
        if drawdown_from_peak >= TRAILING_STOP:
            return True, 'trailing_stop', return_pct
        
        # Exit condition 4: MA crossunder (only after minimum hold)
        if self.hold_days >= MIN_HOLD_DAYS and ma_crossunder:
            return True, 'ma_crossunder', return_pct
        
        return False, None, return_pct

# Portfolio tracking
cash = INITIAL_CAPITAL
positions = {}
portfolio_value = []
trade_log = []
dates = sorted(set([date for ticker in data for date in data[ticker].index]))

print("\n🚀 Running backtest...")
print(f"Period: {dates[0].date()} to {dates[-1].date()}")

# ============================
# MAIN BACKTEST LOOP
# ============================
for current_date in dates:
    # Check exits first
    positions_to_close = []
    
    for ticker, pos in positions.items():
        if current_date not in data[ticker].index:
            continue
        
        current_price = data[ticker].loc[current_date, 'Close']
        ma_crossunder = data[ticker].loc[current_date, 'MA_crossunder'] == 1
        
        should_exit, exit_reason, return_pct = pos.check_exit(
            current_date, current_price, ma_crossunder
        )
        
        if should_exit:
            positions_to_close.append((ticker, exit_reason))
    
    # Close positions
    for ticker, exit_reason in positions_to_close:
        pos = positions[ticker]
        exit_price = data[ticker].loc[current_date, 'Close']
        exit_value = pos.shares * exit_price
        pnl = exit_value - (pos.shares * pos.entry_price)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        
        cash += exit_value
        
        trade_log.append({
            'ticker': ticker,
            'entry_date': pos.entry_date,
            'exit_date': current_date,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'shares': pos.shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': exit_reason,
            'hold_days': pos.hold_days,
            'peak_price': pos.peak_price,
            'max_gain_pct': (pos.peak_price - pos.entry_price) / pos.entry_price * 100
        })
        
        del positions[ticker]
    
    # Check for new entries (if we have room)
    if len(positions) < MAX_POSITIONS:
        signals = []
        
        for ticker in data:
            if ticker in positions:
                continue
            
            if current_date not in data[ticker].index:
                continue
            
            row = data[ticker].loc[current_date]
            
            # Entry conditions
            if (row['MA_cross'] == 1 and 
                row['Volume_ratio'] >= VOLUME_THRESHOLD and
                not pd.isna(row['Momentum'])):
                
                signals.append({
                    'ticker': ticker,
                    'momentum': row['Momentum'],
                    'price': row['Close']
                })
        
        # Sort by momentum and take top signals
        if signals:
            signals = sorted(signals, key=lambda x: x['momentum'], reverse=True)
            slots_available = MAX_POSITIONS - len(positions)
            
            for signal in signals[:slots_available]:
                position_value = INITIAL_CAPITAL * POSITION_SIZE
                shares = int(position_value / signal['price'])
                
                if shares > 0 and cash >= shares * signal['price']:
                    cost = shares * signal['price']
                    cash -= cost
                    
                    positions[signal['ticker']] = Position(
                        ticker=signal['ticker'],
                        entry_date=current_date,
                        entry_price=signal['price'],
                        shares=shares
                    )
    
    # Calculate portfolio value
    position_value = sum(
        pos.shares * data[ticker].loc[current_date, 'Close']
        for ticker, pos in positions.items()
        if current_date in data[ticker].index
    )
    total_value = cash + position_value
    portfolio_value.append({
        'date': current_date,
        'value': total_value,
        'cash': cash,
        'positions': len(positions)
    })

# ============================
# RESULTS ANALYSIS
# ============================
portfolio_df = pd.DataFrame(portfolio_value)
portfolio_df.set_index('date', inplace=True)

final_value = portfolio_df['value'].iloc[-1]
total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
trades_df = pd.DataFrame(trade_log)

print("\n" + "=" * 80)
print("📊 V4.0 RESULTS - SMART EXITS")
print("=" * 80)

print(f"\n💰 Portfolio Performance:")
print(f"  Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
print(f"  Final Value:      ${final_value:,.2f}")
print(f"  Total Return:     {total_return*100:.2f}%")
print(f"  Total Trades:     {len(trades_df)}")

if len(trades_df) > 0:
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]
    
    print(f"\n📈 Trade Statistics:")
    print(f"  Winning Trades:   {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
    print(f"  Losing Trades:    {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
    print(f"  Avg Winner:       {winners['pnl_pct'].mean():.2f}%")
    print(f"  Avg Loser:        {losers['pnl_pct'].mean():.2f}%")
    print(f"  Best Trade:       {trades_df['pnl_pct'].max():.2f}% ({trades_df.loc[trades_df['pnl_pct'].idxmax(), 'ticker']})")
    print(f"  Worst Trade:      {trades_df['pnl_pct'].min():.2f}% ({trades_df.loc[trades_df['pnl_pct'].idxmin(), 'ticker']})")
    print(f"  Avg Hold Days:    {trades_df['hold_days'].mean():.0f} days")
    
    print(f"\n🎯 Exit Reason Breakdown:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / len(trades_df) * 100
        avg_return = trades_df[trades_df['exit_reason'] == reason]['pnl_pct'].mean()
        print(f"  {reason:15s}: {count:2d} trades ({pct:4.1f}%) - Avg: {avg_return:+6.2f}%")
    
    print(f"\n🏔️  Peak Analysis (Trailing Stop Impact):")
    print(f"  Avg Max Gain Reached:  {trades_df['max_gain_pct'].mean():.2f}%")
    print(f"  Avg Actual Return:     {trades_df['pnl_pct'].mean():.2f}%")
    print(f"  Avg Giveback:          {(trades_df['max_gain_pct'] - trades_df['pnl_pct']).mean():.2f}%")

# Benchmark comparison
print(f"\n📊 Buy & Hold Benchmark:")
for ticker in ['SPY', 'QQQ', 'NVDA']:
    try:
        bench = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        bench_return = (bench['Close'].iloc[-1] - bench['Close'].iloc[0]) / bench['Close'].iloc[0]
        print(f"  {ticker:6s}: {bench_return*100:+.2f}%")
    except:
        pass

# Calculate returns
portfolio_df['returns'] = portfolio_df['value'].pct_change()
sharpe = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252)
max_dd = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1).min()

print(f"\n📉 Risk Metrics:")
print(f"  Sharpe Ratio:     {sharpe:.2f}")
print(f"  Max Drawdown:     {max_dd*100:.2f}%")

# ============================
# SAVE RESULTS
# ============================
results_dir = Path('results/moving_average_v4')
results_dir.mkdir(parents=True, exist_ok=True)

trades_df.to_csv(results_dir / 'trade_log.csv', index=False)
portfolio_df.to_csv(results_dir / 'portfolio_value.csv')

print(f"\n💾 Results saved to: {results_dir}/")

# ============================
# VISUALIZATIONS
# ============================
print("\n📈 Generating charts...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Portfolio Value
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(portfolio_df.index, portfolio_df['value'], label='Strategy v4.0', linewidth=2)

# Add buy & hold comparison
spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
spy_norm = (spy['Close'] / spy['Close'].iloc[0]) * INITIAL_CAPITAL
ax1.plot(spy.index, spy_norm, label='Buy & Hold (SPY)', alpha=0.7, linestyle='--')

ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Value ($)')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 2. Cumulative Returns
ax2 = fig.add_subplot(gs[1, 0])
cum_returns = (portfolio_df['value'] / INITIAL_CAPITAL - 1) * 100
spy_cum = (spy_norm / INITIAL_CAPITAL - 1) * 100
ax2.plot(portfolio_df.index, cum_returns, label='Strategy v4.0', linewidth=2)
ax2.plot(spy.index, spy_cum, label='Buy & Hold (SPY)', alpha=0.7, linestyle='--')
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
ax2.set_ylabel('Return (%)')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Drawdown
ax3 = fig.add_subplot(gs[1, 1])
drawdown = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1) * 100
ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
ax3.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.grid(alpha=0.3)

# 4. Trade Distribution
ax4 = fig.add_subplot(gs[2, 0])
if len(trades_df) > 0:
    ax4.hist(trades_df['pnl_pct'], bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(x=trades_df['pnl_pct'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f"Mean: {trades_df['pnl_pct'].mean():.1f}%")
    ax4.axvline(x=0, color='red', linestyle=':', linewidth=1)
    ax4.set_title('Trade Return Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Number of Trades')
    ax4.legend()
    ax4.grid(alpha=0.3)

# 5. Exit Reasons
ax5 = fig.add_subplot(gs[2, 1])
if len(trades_df) > 0:
    exit_counts = trades_df['exit_reason'].value_counts()
    colors = {'take_profit': 'green', 'trailing_stop': 'orange', 
              'ma_crossunder': 'blue', 'max_hold': 'purple'}
    bars = ax5.bar(range(len(exit_counts)), exit_counts.values, 
                   color=[colors.get(x, 'gray') for x in exit_counts.index])
    ax5.set_xticks(range(len(exit_counts)))
    ax5.set_xticklabels(exit_counts.index, rotation=45, ha='right')
    ax5.set_title('Exit Reasons', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Trades')
    ax5.grid(alpha=0.3, axis='y')
    
    # Add percentages on bars
    for i, (bar, count) in enumerate(zip(bars, exit_counts.values)):
        pct = count / len(trades_df) * 100
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('Moving Average Strategy v4.0 - SMART EXITS', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(results_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Chart saved: {results_dir}/performance_summary.png")

print("\n" + "=" * 80)
print("🎯 V4.0 Complete! Check the results above.")
print("=" * 80)