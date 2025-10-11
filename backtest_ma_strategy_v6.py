import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# V6 STRATEGY CONFIGURATION
# ==========================================
# Universe: 4 Non-Tech Large Caps
STOCKS = ['JPM', 'UNH', 'XOM', 'WMT']

# Portfolio rules
POSITIONS = 4
WEIGHT_PER_POSITION = 0.25  # 25% each

# Rebalance frequency
REBALANCE_DAYS = 180

# Exit rules
MAX_HOLD_DAYS = 365
TAKE_PROFIT_PCT = 0.50  # +50%

# Backtest period
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# Initial capital
INITIAL_CAPITAL = 100000

# Create results directory
RESULTS_DIR = Path('results/v6_nontech_equalweight')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎯 V6 STRATEGY - NON-TECH LARGE CAPS (EQUAL WEIGHT)")
print("=" * 80)
print(f"Universe: {STOCKS}")
print(f"Position size: {WEIGHT_PER_POSITION*100}% each ({POSITIONS} positions)")
print(f"Rebalance: Every {REBALANCE_DAYS} days")
print(f"Exit rules: {MAX_HOLD_DAYS} days max hold OR +{TAKE_PROFIT_PCT*100}% take-profit")
print(f"Period: {START_DATE} to {END_DATE}")
print("=" * 80)

# ==========================================
# DOWNLOAD DATA
# ==========================================
print("\n📊 Downloading price data...")
raw_data = yf.download(STOCKS + ['SPY', 'QQQ'], start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)

# Handle both single and multi-level column structures
if isinstance(raw_data.columns, pd.MultiIndex):
    data = raw_data['Adj Close']
else:
    data = raw_data

data = data.ffill()

print(f"✓ Downloaded {len(data)} days of data")
print(f"✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")

# ==========================================
# BACKTEST ENGINE
# ==========================================
class Position:
    def __init__(self, ticker, entry_date, entry_price, shares, position_value):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.position_value = position_value
        self.days_held = 0
        self.exit_price = None
        self.exit_date = None
        
    def update(self, current_price, current_date):
        self.days_held = (current_date - self.entry_date).days
        current_value = self.shares * current_price
        return_pct = (current_price / self.entry_price - 1)
        return current_value, return_pct

# Initialize
cash = INITIAL_CAPITAL
positions = {}
portfolio_values = []
trade_log = []
completed_trades = []  # For CSV export
last_rebalance_date = None

# Track for rebalancing
entry_dates = {}

print("\n🚀 Running backtest...\n")

for i, date in enumerate(data.index):
    current_date = date
    
    # ==========================================
    # INITIAL ENTRY (Day 1)
    # ==========================================
    if i == 0:
        print(f"📅 {date.date()} - INITIAL ENTRY")
        position_size = INITIAL_CAPITAL * WEIGHT_PER_POSITION
        
        for stock in STOCKS:
            price = data.loc[date, stock]
            shares = position_size / price
            
            positions[stock] = Position(
                ticker=stock,
                entry_date=date,
                entry_price=price,
                shares=shares,
                position_value=position_size
            )
            
            entry_dates[stock] = date
            cash -= position_size
            
            trade_log.append({
                'Date': date,
                'Action': 'BUY',
                'Ticker': stock,
                'Price': price,
                'Shares': shares,
                'Value': position_size,
                'Reason': 'Initial Entry'
            })
            
            print(f"  ✓ BUY {stock}: {shares:.2f} shares @ ${price:.2f} = ${position_size:,.2f}")
        
        last_rebalance_date = date
        print(f"  💰 Cash remaining: ${cash:,.2f}\n")
    
    # ==========================================
    # DAILY POSITION MANAGEMENT
    # ==========================================
    portfolio_value = cash
    
    # Check each position for exit conditions
    positions_to_exit = []
    
    for stock, pos in positions.items():
        price = data.loc[date, stock]
        current_value, return_pct = pos.update(price, date)
        portfolio_value += current_value
        
        # Exit Rule 1: Max hold period (365 days)
        if pos.days_held >= MAX_HOLD_DAYS:
            positions_to_exit.append((stock, pos, 'max_hold'))
        
        # Exit Rule 2: Take profit (+50%)
        elif return_pct >= TAKE_PROFIT_PCT:
            positions_to_exit.append((stock, pos, 'take_profit'))
    
    # Execute exits
    for stock, pos, reason in positions_to_exit:
        exit_price = data.loc[date, stock]
        exit_value = pos.shares * exit_price
        profit = exit_value - pos.position_value
        return_pct = (exit_price / pos.entry_price - 1) * 100
        
        cash += exit_value
        
        # Record completed trade for CSV
        completed_trades.append({
            'ticker': stock,
            'entry_date': pos.entry_date.strftime('%Y-%m-%d'),
            'exit_date': date.strftime('%Y-%m-%d'),
            'entry_price': round(pos.entry_price, 2),
            'exit_price': round(exit_price, 2),
            'shares': round(pos.shares, 2),
            'pnl': round(profit, 2),
            'pnl_pct': round(return_pct, 2),
            'exit_reason': reason,
            'hold_days': pos.days_held
        })
        
        trade_log.append({
            'Date': date,
            'Action': 'SELL',
            'Ticker': stock,
            'Price': exit_price,
            'Shares': pos.shares,
            'Value': exit_value,
            'Reason': reason,
            'Profit': profit,
            'Return%': return_pct,
            'Days_Held': pos.days_held
        })
        
        print(f"📅 {date.date()} - EXIT {stock}")
        print(f"  ✓ SELL {stock}: {pos.shares:.2f} shares @ ${exit_price:.2f}")
        print(f"  📊 Entry: ${pos.entry_price:.2f} | Exit: ${exit_price:.2f}")
        print(f"  💰 P&L: ${profit:,.2f} ({return_pct:+.2f}%) | Held {pos.days_held} days")
        print(f"  🎯 Reason: {reason}\n")
        
        # Re-enter position immediately
        position_size = portfolio_value * WEIGHT_PER_POSITION
        new_shares = position_size / exit_price
        
        positions[stock] = Position(
            ticker=stock,
            entry_date=date,
            entry_price=exit_price,
            shares=new_shares,
            position_value=position_size
        )
        
        entry_dates[stock] = date
        cash -= position_size
        
        trade_log.append({
            'Date': date,
            'Action': 'BUY',
            'Ticker': stock,
            'Price': exit_price,
            'Shares': new_shares,
            'Value': position_size,
            'Reason': 'Re-entry after exit'
        })
        
        print(f"  ✓ RE-ENTER {stock}: {new_shares:.2f} shares @ ${exit_price:.2f} = ${position_size:,.2f}")
        print(f"  💰 Cash after re-entry: ${cash:,.2f}\n")
    
    # ==========================================
    # REBALANCE CHECK (Every 180 days)
    # ==========================================
    if last_rebalance_date is not None:
        days_since_rebalance = (date - last_rebalance_date).days
        
        if days_since_rebalance >= REBALANCE_DAYS:
            print(f"📅 {date.date()} - REBALANCE (180 days since last rebalance)")
            
            # Calculate current portfolio value
            portfolio_value = cash
            for stock, pos in positions.items():
                price = data.loc[date, stock]
                portfolio_value += pos.shares * price
            
            # Rebalance to 25% each
            target_value = portfolio_value * WEIGHT_PER_POSITION
            
            for stock in STOCKS:
                if stock in positions:
                    pos = positions[stock]
                    current_price = data.loc[date, stock]
                    current_value = pos.shares * current_price
                    
                    # Calculate difference
                    diff_value = target_value - current_value
                    
                    if abs(diff_value) > 100:  # Only rebalance if difference > $100
                        if diff_value > 0:
                            # Buy more
                            shares_to_buy = diff_value / current_price
                            positions[stock].shares += shares_to_buy
                            cash -= diff_value
                            
                            trade_log.append({
                                'Date': date,
                                'Action': 'BUY',
                                'Ticker': stock,
                                'Price': current_price,
                                'Shares': shares_to_buy,
                                'Value': diff_value,
                                'Reason': 'Rebalance'
                            })
                            
                            print(f"  ✓ BUY {stock}: {shares_to_buy:.2f} shares @ ${current_price:.2f} = ${diff_value:,.2f}")
                        else:
                            # Sell some
                            shares_to_sell = abs(diff_value) / current_price
                            positions[stock].shares -= shares_to_sell
                            cash += abs(diff_value)
                            
                            trade_log.append({
                                'Date': date,
                                'Action': 'SELL',
                                'Ticker': stock,
                                'Price': current_price,
                                'Shares': shares_to_sell,
                                'Value': abs(diff_value),
                                'Reason': 'Rebalance'
                            })
                            
                            print(f"  ✓ SELL {stock}: {shares_to_sell:.2f} shares @ ${current_price:.2f} = ${abs(diff_value):,.2f}")
            
            last_rebalance_date = date
            print(f"  💰 Cash after rebalance: ${cash:,.2f}\n")
    
    # Record portfolio value
    portfolio_values.append({
        'Date': date,
        'Portfolio_Value': portfolio_value
    })

# ==========================================
# RESULTS CALCULATION
# ==========================================
print("\n" + "=" * 80)
print("📊 CALCULATING RESULTS...")
print("=" * 80)

# Convert to DataFrames
portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')
trades_df = pd.DataFrame(trade_log)

# Calculate returns
portfolio_df['V6_Return'] = (portfolio_df['Portfolio_Value'] / INITIAL_CAPITAL - 1) * 100
portfolio_df['SPY_Return'] = (data['SPY'] / data['SPY'].iloc[0] - 1) * 100
portfolio_df['QQQ_Return'] = (data['QQQ'] / data['QQQ'].iloc[0] - 1) * 100

# Final results
final_value = portfolio_df['Portfolio_Value'].iloc[-1]
total_return = portfolio_df['V6_Return'].iloc[-1]
spy_return = portfolio_df['SPY_Return'].iloc[-1]
qqq_return = portfolio_df['QQQ_Return'].iloc[-1]

print(f"\n💰 FINAL RESULTS:")
print(f"{'='*80}")
print(f"Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value:        ${final_value:,.2f}")
print(f"Total Return:       {total_return:+.2f}%")
print(f"{'='*80}")
print(f"SPY Return:         {spy_return:+.2f}%")
print(f"QQQ Return:         {qqq_return:+.2f}%")
print(f"{'='*80}")
print(f"V6 vs SPY:          {total_return - spy_return:+.2f}%")
print(f"V6 vs QQQ:          {total_return - qqq_return:+.2f}%")
print(f"{'='*80}")

# Trade statistics
buy_trades = trades_df[trades_df['Action'] == 'BUY']
sell_trades = trades_df[(trades_df['Action'] == 'SELL') & (trades_df['Reason'] != 'Rebalance')]

print(f"\n📈 TRADE STATISTICS:")
print(f"{'='*80}")
print(f"Total Trades:       {len(trades_df)}")
print(f"Buy Trades:         {len(buy_trades)}")
print(f"Exit Trades:        {len(sell_trades)}")

if len(sell_trades) > 0:
    winning_trades = sell_trades[sell_trades['Return%'] > 0]
    win_rate = len(winning_trades) / len(sell_trades) * 100
    avg_return = sell_trades['Return%'].mean()
    avg_hold = sell_trades['Days_Held'].mean()
    
    print(f"Winning Trades:     {len(winning_trades)}/{len(sell_trades)} ({win_rate:.1f}%)")
    print(f"Avg Return:         {avg_return:+.2f}%")
    print(f"Avg Hold Period:    {avg_hold:.0f} days")

print(f"{'='*80}")

# ==========================================
# SAVE RESULTS
# ==========================================
# Save completed trades CSV
if completed_trades:
    completed_df = pd.DataFrame(completed_trades)
    csv_filename = RESULTS_DIR / 'trade_log.csv'
    completed_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Trade log saved: {csv_filename}")

# Save portfolio values
portfolio_df.to_csv(RESULTS_DIR / 'portfolio_value.csv')
print(f"✓ Portfolio values saved: {RESULTS_DIR / 'portfolio_value.csv'}")

# ==========================================
# PERFORMANCE CHART
# ==========================================
print("\n📈 Generating charts...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Performance comparison
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(portfolio_df.index, portfolio_df['V6_Return'], label='V6 Strategy', linewidth=2.5, color='#10b981')
ax1.plot(portfolio_df.index, portfolio_df['SPY_Return'], label='SPY', linewidth=2, color='#ef4444', linestyle='--', alpha=0.7)
ax1.plot(portfolio_df.index, portfolio_df['QQQ_Return'], label='QQQ', linewidth=2, color='#3b82f6', linestyle=':', alpha=0.7)
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_title('V6 Strategy: Non-Tech Large Caps (Equal Weight) vs Benchmarks', fontsize=14, fontweight='bold')
ax1.set_ylabel('Return (%)', fontsize=12)
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Portfolio value
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(portfolio_df.index, portfolio_df['Portfolio_Value'], linewidth=2.5, color='#10b981')
ax2.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
ax2.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value ($)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 3. Drawdown
ax3 = fig.add_subplot(gs[1, 1])
drawdown = (portfolio_df['Portfolio_Value'] / portfolio_df['Portfolio_Value'].cummax() - 1) * 100
ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
ax3.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
ax3.set_ylabel('Drawdown (%)', fontsize=12)
ax3.grid(alpha=0.3)

# 4. Trade returns (if any exits)
ax4 = fig.add_subplot(gs[2, 0])
if completed_trades:
    completed_df = pd.DataFrame(completed_trades)
    colors = ['green' if x > 0 else 'red' for x in completed_df['pnl_pct']]
    ax4.bar(range(len(completed_df)), completed_df['pnl_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axhline(y=completed_df['pnl_pct'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f"Mean: {completed_df['pnl_pct'].mean():.1f}%")
    ax4.set_title('Exit Trade Returns', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade Number', fontsize=10)
    ax4.set_ylabel('Return (%)', fontsize=10)
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, 'No exits yet', ha='center', va='center', fontsize=14)
    ax4.set_title('Exit Trade Returns', fontsize=12, fontweight='bold')

# 5. Performance comparison bars
ax5 = fig.add_subplot(gs[2, 1])
comparison_data = {
    'V6 Strategy': total_return,
    'SPY': spy_return,
    'QQQ': qqq_return
}
colors_map = {'V6 Strategy': '#10b981', 'SPY': '#ef4444', 'QQQ': '#3b82f6'}
bars = ax5.bar(range(len(comparison_data)), list(comparison_data.values()), 
               color=[colors_map[k] for k in comparison_data.keys()])
ax5.set_xticks(range(len(comparison_data)))
ax5.set_xticklabels(list(comparison_data.keys()), rotation=0)
ax5.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax5.set_ylabel('Return (%)', fontsize=10)
ax5.grid(alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add percentages on bars
for i, (bar, val) in enumerate(zip(bars, comparison_data.values())):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if val > 0 else -5),
            f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
            fontsize=10, fontweight='bold')

plt.suptitle('V6 Strategy: Non-Tech Large Caps - Equal Weight', 
             fontsize=16, fontweight='bold', y=0.995)

chart_filename = RESULTS_DIR / 'performance_summary.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
print(f"✓ Performance chart saved: {chart_filename}")

print(f"\n{'='*80}")
print("✅ V6 Backtest complete!")
print(f"{'='*80}")
print(f"📁 Results saved to: {RESULTS_DIR}/")
print(f"   - trade_log.csv")
print(f"   - portfolio_value.csv")
print(f"   - performance_summary.png")
print(f"{'='*80}")