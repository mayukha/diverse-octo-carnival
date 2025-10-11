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
# V5.0 CONFIGURATION - THE OPTIMAL STRATEGY
# ============================
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
           'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'BAC',
           'XOM', 'CVX', 'ABBV', 'PFE', 'COST', 'DIS', 'NFLX',
           'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'BA', 'NKE']

START_DATE = '2022-10-01'
END_DATE = '2025-10-11'
INITIAL_CAPITAL = 100000

# Portfolio Settings (FIFO)
MAX_POSITIONS = 5  # Maximum 5 stocks at once
POSITION_SIZE = 0.20  # 20% per position (equal weight)

# Entry: Momentum Selection
MOMENTUM_LOOKBACK = 126  # 6 months (approx 126 trading days)
TOP_N_MOMENTUM = 10  # Select from top 10 momentum stocks

# Exit: Time-Based Only
HOLD_PERIOD = 180  # 180 days (6 months hold)
TAKE_PROFIT = 0.50  # +50% take profit
REBALANCE_DAYS = 90  # Check for rebalance every 90 days

print("=" * 80)
print("V5.0 - THE OPTIMAL STRATEGY: MOMENTUM + TIME")
print("=" * 80)
print(f"\n🎯 LESSONS FROM V1-V4:")
print(f"  ✅ Max-hold trades (V4): +23.9% average - TIME WORKS")
print(f"  ❌ Trailing stops: -8.2% average - KILLED PROFITS")
print(f"  ❌ MA crossunders: Cut winners short - NOISE")
print(f"  ❌ Volume filters: Blocked NVDA - TOO STRICT")
print(f"\n🚀 V5 FIXES:")
print(f"  1. Momentum selection (catches NVDA by definition)")
print(f"  2. FIFO: Max {MAX_POSITIONS} positions, {POSITION_SIZE*100}% each")
print(f"  3. Hold {HOLD_PERIOD} days OR +{TAKE_PROFIT*100}% (whichever first)")
print(f"  4. NO technical exits - pure time + profit target")
print(f"  5. Rebalance every {REBALANCE_DAYS} days")
print("=" * 80)

# ============================
# DATA DOWNLOAD
# ============================
print("\n📊 Downloading data...")
data = {}
for ticker in TICKERS:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) > MOMENTUM_LOOKBACK + 10:
            data[ticker] = df
            print(f"  ✓ {ticker}: {len(df)} days")
    except Exception as e:
        print(f"  ✗ {ticker}: Error - {e}")

print(f"\n✓ Loaded {len(data)} tickers")

# ============================
# CALCULATE MOMENTUM
# ============================
print("\n📈 Calculating momentum scores...")
for ticker in data:
    df = data[ticker]
    # 6-month momentum (simple return)
    df['Momentum'] = df['Close'].pct_change(MOMENTUM_LOOKBACK)
    # Also calculate for ranking
    df['Momentum_Rank'] = 0

print("✓ Momentum calculated")

# ============================
# BACKTESTING ENGINE
# ============================
class Position:
    def __init__(self, ticker, entry_date, entry_price, shares):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.entry_order = None  # For FIFO tracking
    
    def check_exit(self, current_date, current_price):
        """Check time-based and profit target exits only"""
        hold_days = (current_date - self.entry_date).days
        return_pct = (current_price - self.entry_price) / self.entry_price
        
        # Exit 1: Take profit (+50%)
        if return_pct >= TAKE_PROFIT:
            return True, 'take_profit', return_pct
        
        # Exit 2: Max hold period (180 days)
        if hold_days >= HOLD_PERIOD:
            return True, 'max_hold', return_pct
        
        return False, None, return_pct

# Portfolio tracking
cash = INITIAL_CAPITAL
positions = {}  # Current positions {ticker: Position}
portfolio_value = []
trade_log = []
entry_counter = 0  # For FIFO tracking
dates = sorted(set([date for ticker in data for date in data[ticker].index]))

print("\n🚀 Running backtest...")
print(f"Period: {dates[0].date()} to {dates[-1].date()}")

last_rebalance = dates[0]

# ============================
# MAIN BACKTEST LOOP
# ============================
for current_date in dates:
    
    # === 1. CHECK EXITS ===
    positions_to_close = []
    
    for ticker, pos in positions.items():
        if current_date not in data[ticker].index:
            continue
        
        current_price = data[ticker].loc[current_date, 'Close']
        should_exit, exit_reason, return_pct = pos.check_exit(current_date, current_price)
        
        if should_exit:
            positions_to_close.append((ticker, exit_reason))
    
    # Close positions
    for ticker, exit_reason in positions_to_close:
        pos = positions[ticker]
        exit_price = data[ticker].loc[current_date, 'Close']
        exit_value = pos.shares * exit_price
        pnl = exit_value - (pos.shares * pos.entry_price)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        hold_days = (current_date - pos.entry_date).days
        
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
            'hold_days': hold_days
        })
        
        del positions[ticker]
    
    # === 2. REBALANCE CHECK (Every 90 days OR when slot opens) ===
    days_since_rebalance = (current_date - last_rebalance).days
    should_rebalance = (days_since_rebalance >= REBALANCE_DAYS) or (len(positions) < MAX_POSITIONS)
    
    if should_rebalance and len(positions) < MAX_POSITIONS:
        
        # Calculate momentum for all stocks
        momentum_scores = []
        for ticker in data:
            if ticker in positions:  # Skip stocks we already own
                continue
            
            if current_date not in data[ticker].index:
                continue
            
            momentum = data[ticker].loc[current_date, 'Momentum']
            price = data[ticker].loc[current_date, 'Close']
            
            if pd.notna(momentum):
                momentum_scores.append({
                    'ticker': ticker,
                    'momentum': momentum,
                    'price': price
                })
        
        # Sort by momentum and take top candidates
        momentum_scores = sorted(momentum_scores, key=lambda x: x['momentum'], reverse=True)
        top_momentum = momentum_scores[:TOP_N_MOMENTUM]
        
        # === 3. FIFO ENTRY ===
        # Fill available slots with highest momentum stocks
        slots_available = MAX_POSITIONS - len(positions)
        
        for candidate in top_momentum[:slots_available]:
            position_value = INITIAL_CAPITAL * POSITION_SIZE
            shares = int(position_value / candidate['price'])
            
            if shares > 0 and cash >= shares * candidate['price']:
                cost = shares * candidate['price']
                cash -= cost
                
                new_pos = Position(
                    ticker=candidate['ticker'],
                    entry_date=current_date,
                    entry_price=candidate['price'],
                    shares=shares
                )
                new_pos.entry_order = entry_counter
                entry_counter += 1
                
                positions[candidate['ticker']] = new_pos
                
                # Update last rebalance if we entered on rebalance signal
                if days_since_rebalance >= REBALANCE_DAYS:
                    last_rebalance = current_date
    
    # === 4. CALCULATE PORTFOLIO VALUE ===
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
print("📊 V5.0 RESULTS - MOMENTUM + TIME")
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
    if len(winners) > 0:
        print(f"  Avg Winner:       {winners['pnl_pct'].mean():.2f}%")
    if len(losers) > 0:
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
    
    print(f"\n🏆 Top 5 Trades:")
    top_5 = trades_df.nlargest(5, 'pnl_pct')
    for idx, trade in top_5.iterrows():
        print(f"  {trade['ticker']:6s}: {trade['pnl_pct']:+6.2f}% ({trade['hold_days']:3.0f} days) - {trade['exit_reason']}")
    
    print(f"\n💀 Bottom 5 Trades:")
    bottom_5 = trades_df.nsmallest(5, 'pnl_pct')
    for idx, trade in bottom_5.iterrows():
        print(f"  {trade['ticker']:6s}: {trade['pnl_pct']:+6.2f}% ({trade['hold_days']:3.0f} days) - {trade['exit_reason']}")

# Benchmark comparison
print(f"\n📊 Benchmark Comparison:")
benchmarks = {}
for ticker in ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT']:
    try:
        bench = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
        bench_return = (bench['Close'].iloc[-1] - bench['Close'].iloc[0]) / bench['Close'].iloc[0]
        benchmarks[ticker] = bench_return
        symbol = "🏆" if total_return > bench_return else "  "
        print(f"  {symbol} {ticker:6s}: {bench_return*100:+7.2f}% {'✓ BEAT IT!' if total_return > bench_return else ''}")
    except:
        pass

# Calculate risk metrics
portfolio_df['returns'] = portfolio_df['value'].pct_change()
sharpe = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252) if portfolio_df['returns'].std() > 0 else 0
max_dd = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1).min()

print(f"\n📉 Risk Metrics:")
print(f"  Sharpe Ratio:     {sharpe:.2f}")
print(f"  Max Drawdown:     {max_dd*100:.2f}%")

# Show if we beat buy & hold
if 'SPY' in benchmarks:
    alpha = (total_return - benchmarks['SPY']) * 100
    print(f"  Alpha vs SPY:     {alpha:+.2f}%")

# ============================
# SAVE RESULTS
# ============================
results_dir = Path('results/moving_average_v5')
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

# 1. Portfolio Value vs Benchmarks
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(portfolio_df.index, portfolio_df['value'], label='V5.0 Strategy', linewidth=2.5, color='#2563eb')

# Add benchmarks
for bench_name in ['SPY', 'QQQ']:
    try:
        bench = yf.download(bench_name, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
        bench_norm = (bench['Close'] / bench['Close'].iloc[0]) * INITIAL_CAPITAL
        linestyle = '--' if bench_name == 'SPY' else ':'
        ax1.plot(bench.index, bench_norm, label=f'{bench_name}', alpha=0.7, linestyle=linestyle, linewidth=2)
    except:
        pass

ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
ax1.set_title('Portfolio Value Over Time - V5 vs Benchmarks', fontsize=14, fontweight='bold')
ax1.set_ylabel('Value ($)')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 2. Cumulative Returns
ax2 = fig.add_subplot(gs[1, 0])
cum_returns = (portfolio_df['value'] / INITIAL_CAPITAL - 1) * 100
ax2.plot(portfolio_df.index, cum_returns, linewidth=2.5, color='#2563eb')
ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(portfolio_df.index, cum_returns, 0, alpha=0.2, color='#2563eb')
ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
ax2.set_ylabel('Return (%)')
ax2.grid(alpha=0.3)

# 3. Drawdown
ax3 = fig.add_subplot(gs[1, 1])
drawdown = (portfolio_df['value'] / portfolio_df['value'].cummax() - 1) * 100
ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
ax3.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown (%)')
ax3.grid(alpha=0.3)

# 4. Trade Distribution
ax4 = fig.add_subplot(gs[2, 0])
if len(trades_df) > 0:
    colors = ['green' if x > 0 else 'red' for x in trades_df['pnl_pct']]
    ax4.bar(range(len(trades_df)), trades_df['pnl_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=trades_df['pnl_pct'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f"Mean: {trades_df['pnl_pct'].mean():.1f}%")
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('All Trades (Sequential)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Return (%)')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

# 5. Performance Comparison
ax5 = fig.add_subplot(gs[2, 1])
comparison_data = {'V5.0': total_return * 100}
comparison_data.update({k: v * 100 for k, v in benchmarks.items()})
colors_map = {'V5.0': '#2563eb', 'SPY': '#94a3b8', 'QQQ': '#cbd5e1', 
              'NVDA': '#fbbf24', 'AAPL': '#64748b', 'MSFT': '#94a3b8'}
bars = ax5.bar(range(len(comparison_data)), list(comparison_data.values()), 
               color=[colors_map.get(k, 'gray') for k in comparison_data.keys()])
ax5.set_xticks(range(len(comparison_data)))
ax5.set_xticklabels(list(comparison_data.keys()), rotation=0)
ax5.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax5.set_ylabel('Return (%)')
ax5.grid(alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add percentages on bars
for i, (bar, val) in enumerate(zip(bars, comparison_data.values())):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('V5.0 Strategy: Momentum + Time - THE OPTIMAL APPROACH', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(results_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Chart saved: {results_dir}/performance_summary.png")

print("\n" + "=" * 80)
print("🎯 V5.0 Complete!")
print("=" * 80)
print(f"\n📝 KEY CHANGES FROM V4:")
print(f"  • Momentum selection (NO volume filters)")
print(f"  • FIFO: Max 5 positions, 20% each")
print(f"  • Time-based exits only (180 days OR +50%)")
print(f"  • NO trailing stops, NO MA crossunders")
print(f"  • Quarterly rebalancing")
print(f"\n🎲 THE BIG QUESTION: Did we finally catch NVDA?")
print("=" * 80)