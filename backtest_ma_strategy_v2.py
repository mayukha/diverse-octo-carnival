"""
Moving Average Strategy - VERSION 3.0 - AGGRESSIVE REWRITE
Key Changes:
- Back to 20/50 MA (50/200 was too slow)
- REMOVED stop-loss entirely (was killing trades prematurely)
- Keep take-profit at 50%
- Volume threshold at 1.1x
- NO correlation filter
- NO drawdown filter
- Allow up to 7 positions (70% capital utilization)
- Smaller position size (12% per position)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# ==================== CONFIGURATION ====================
TICKERS = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT', 'DIS', 'BA', 'PG', 'V', 'NVDA']
START_DATE = '2022-10-10'
END_DATE = '2025-10-10'
INITIAL_CAPITAL = 100000

# VERSION 3 PARAMETERS - AGGRESSIVE
FAST_MA = 20
SLOW_MA = 50
VOLUME_MULT = 1.1
STOP_LOSS = None  # REMOVED - Let winners run, exit only on MA cross
TAKE_PROFIT = 0.50
MAX_POSITIONS = 7  # Increased from 5
POSITION_SIZE = 0.12  # 12% per position (allows 7-8 positions)

# ==================== DATA DOWNLOAD ====================
def download_data(tickers, start, end):
    """Download historical data for all tickers"""
    print(f"\nDownloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, end=end, progress=False)
    
    if len(tickers) == 1:
        data = pd.DataFrame({
            'Close': data['Close'],
            'Volume': data['Volume']
        })
        data.columns = pd.MultiIndex.from_product([[tickers[0]], data.columns])
    
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Total trading days: {len(data)}")
    return data

# ==================== INDICATOR CALCULATION ====================
def calculate_indicators(data, tickers):
    """Calculate technical indicators for all stocks"""
    print("\nCalculating indicators...")
    indicators = {}
    
    for ticker in tickers:
        df = pd.DataFrame()
        df['close'] = data['Close'][ticker]
        df['volume'] = data['Volume'][ticker]
        
        # Moving averages
        df['ma_fast'] = df['close'].rolling(window=FAST_MA).mean()
        df['ma_slow'] = df['close'].rolling(window=SLOW_MA).mean()
        
        # Volume average
        df['volume_avg'] = df['volume'].rolling(window=50).mean()
        
        # Signal: fast MA crosses above slow MA
        df['signal'] = (df['ma_fast'] > df['ma_slow']).astype(int)
        df['prev_signal'] = df['signal'].shift(1)
        df['crossover'] = (df['signal'] == 1) & (df['prev_signal'] == 0)
        
        # Exit signal: fast MA crosses below slow MA
        df['crossunder'] = (df['signal'] == 0) & (df['prev_signal'] == 1)
        
        indicators[ticker] = df
    
    return indicators

# ==================== BACKTEST ENGINE ====================
class Portfolio:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = []
        
    def can_open_position(self):
        return len(self.positions) < MAX_POSITIONS
    
    def open_position(self, ticker, price, date):
        if not self.can_open_position():
            return False
        
        position_value = self.initial_capital * POSITION_SIZE
        shares = position_value // price
        cost = shares * price
        
        if cost > self.cash:
            return False
        
        self.cash -= cost
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'take_profit': price * (1 + TAKE_PROFIT)
        }
        return True
    
    def close_position(self, ticker, price, date, reason):
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        proceeds = pos['shares'] * price
        self.cash += proceeds
        
        cost = pos['shares'] * pos['entry_price']
        pnl = proceeds - cost
        pnl_pct = (price / pos['entry_price'] - 1) * 100
        
        self.trades.append({
            'ticker': ticker,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'hold_days': (date - pos['entry_date']).days
        })
        
        del self.positions[ticker]
    
    def get_total_value(self, current_prices):
        positions_value = sum(
            pos['shares'] * current_prices.get(ticker, pos['entry_price'])
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def check_take_profit(self, ticker, price, date):
        """Check only take-profit (no stop-loss)"""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        if price >= pos['take_profit']:
            self.close_position(ticker, price, date, 'take_profit')

# ==================== MAIN BACKTEST ====================
def run_backtest(data, indicators, tickers):
    print("\nRunning backtest...")
    portfolio = Portfolio(INITIAL_CAPITAL)
    
    dates = data.index[SLOW_MA:]
    
    for date in dates:
        current_prices = {ticker: data['Close'][ticker].loc[date] for ticker in tickers}
        
        # Check existing positions for exits
        for ticker in list(portfolio.positions.keys()):
            price = current_prices[ticker]
            signals = indicators[ticker].loc[date]
            
            # Check take-profit
            portfolio.check_take_profit(ticker, price, date)
            
            # Check MA crossunder (sell signal)
            if ticker in portfolio.positions:
                if signals['crossunder']:
                    portfolio.close_position(ticker, price, date, 'ma_crossunder')
        
        # Look for new entries - PRIORITIZE BY MOMENTUM
        entry_candidates = []
        for ticker in tickers:
            if ticker in portfolio.positions:
                continue
            
            signals = indicators[ticker].loc[date]
            price = current_prices[ticker]
            
            # ENTRY CONDITIONS (SIMPLIFIED)
            if (signals['crossover'] and 
                signals['volume'] >= signals['volume_avg'] * VOLUME_MULT and
                not pd.isna(signals['ma_slow'])):
                
                # Calculate momentum score (% above slow MA)
                momentum = (signals['ma_fast'] - signals['ma_slow']) / signals['ma_slow']
                entry_candidates.append((ticker, price, momentum))
        
        # Enter positions sorted by momentum (strongest first)
        entry_candidates.sort(key=lambda x: x[2], reverse=True)
        for ticker, price, momentum in entry_candidates:
            if not portfolio.can_open_position():
                break
            portfolio.open_position(ticker, price, date)
        
        # Record daily value
        total_value = portfolio.get_total_value(current_prices)
        portfolio.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': portfolio.cash,
            'positions': len(portfolio.positions)
        })
    
    # Close remaining positions
    final_date = dates[-1]
    for ticker in list(portfolio.positions.keys()):
        price = current_prices[ticker]
        portfolio.close_position(ticker, price, final_date, 'final_close')
    
    return portfolio

# ==================== PERFORMANCE ANALYSIS ====================
def calculate_metrics(portfolio, data, tickers):
    """Calculate performance metrics"""
    df_values = pd.DataFrame(portfolio.daily_values).set_index('date')
    
    total_return = (df_values['total_value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    
    # Benchmark
    benchmark_returns = []
    for ticker in tickers:
        start_price = data['Close'][ticker].iloc[0]
        end_price = data['Close'][ticker].iloc[-1]
        ticker_return = (end_price / start_price - 1) * 100
        benchmark_returns.append(ticker_return)
    
    benchmark_return = np.mean(benchmark_returns)
    alpha = total_return - benchmark_return
    
    # Risk metrics
    daily_returns = df_values['total_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Trade statistics
    trades_df = pd.DataFrame(portfolio.trades)
    winning_trades = trades_df[trades_df['pnl'] > 0] if len(trades_df) > 0 else pd.DataFrame()
    losing_trades = trades_df[trades_df['pnl'] <= 0] if len(trades_df) > 0 else pd.DataFrame()
    
    return {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'alpha': alpha,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
        'df_values': df_values,
        'trades_df': trades_df
    }

# ==================== VISUALIZATION ====================
def plot_results(metrics, data, tickers):
    """Create performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Moving Average Strategy v3.0 - NO STOP-LOSS', fontsize=16, fontweight='bold')
    
    df_values = metrics['df_values']
    
    # 1. Portfolio value
    ax1 = axes[0, 0]
    ax1.plot(df_values.index, df_values['total_value'], label='Strategy v3.0', linewidth=2)
    
    benchmark_values = [INITIAL_CAPITAL]
    for i in range(1, len(data)):
        daily_returns = [data['Close'][ticker].iloc[i] / data['Close'][ticker].iloc[i-1] - 1 
                        for ticker in tickers]
        avg_return = np.mean(daily_returns)
        benchmark_values.append(benchmark_values[-1] * (1 + avg_return))
    
    benchmark_dates = data.index[:len(benchmark_values)]
    ax1.plot(benchmark_dates, benchmark_values, label='Buy & Hold', 
             linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # 2. Cumulative returns
    ax2 = axes[0, 1]
    strategy_returns = (df_values['total_value'] / INITIAL_CAPITAL - 1) * 100
    benchmark_returns_series = (pd.Series(benchmark_values, index=benchmark_dates) / INITIAL_CAPITAL - 1) * 100
    
    ax2.plot(strategy_returns.index, strategy_returns, label='Strategy v3.0', linewidth=2)
    ax2.plot(benchmark_returns_series.index, benchmark_returns_series, 
             label='Buy & Hold', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title('Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[1, 0]
    daily_rets = df_values['total_value'].pct_change()
    cumulative = (1 + daily_rets).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax3.set_title('Portfolio Drawdown')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(alpha=0.3)
    
    # 4. Trade distribution
    ax4 = axes[1, 1]
    trades_df = metrics['trades_df']
    if len(trades_df) > 0:
        ax4.hist(trades_df['pnl_pct'], bins=20, edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(trades_df['pnl_pct'].mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {trades_df["pnl_pct"].mean():.1f}%')
        ax4.set_title('Trade Return Distribution')
        ax4.set_xlabel('Return (%)')
        ax4.set_ylabel('Number of Trades')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('results/moving_average_v3', exist_ok=True)
    plt.savefig('results/moving_average_v3/performance_summary.png', dpi=300, bbox_inches='tight')
    print("\nCharts saved to: results/moving_average_v3/performance_summary.png")
    
    return fig

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    data = download_data(TICKERS, START_DATE, END_DATE)
    indicators = calculate_indicators(data, TICKERS)
    portfolio = run_backtest(data, indicators, TICKERS)
    
    print("\nBacktest complete!")
    print(f"Total trades: {len(portfolio.trades)}")
    print(f"Final positions: {len(portfolio.positions)}")
    
    metrics = calculate_metrics(portfolio, data, TICKERS)
    
    print("\n" + "="*60)
    print("STRATEGY V3.0 - NO STOP-LOSS, LET WINNERS RUN")
    print("="*60)
    print(f"Initial Capital:        ${INITIAL_CAPITAL:,.0f}")
    print(f"Final Value:            ${portfolio.daily_values[-1]['total_value']:,.0f}")
    print(f"Total Return:           {metrics['total_return']:.2f}%")
    print(f"Benchmark (B&H):        {metrics['benchmark_return']:.2f}%")
    print(f"Alpha:                  {metrics['alpha']:.2f}%")
    print(f"\nRisk Metrics:")
    print(f"Max Drawdown:           {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:           {metrics['sharpe']:.3f}")
    print(f"\nTrade Statistics:")
    print(f"Total Trades:           {metrics['total_trades']}")
    print(f"Winning Trades:         {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"Losing Trades:          {metrics['losing_trades']}")
    print(f"Average Win:            {metrics['avg_win']:.2f}%")
    print(f"Average Loss:           {metrics['avg_loss']:.2f}%")
    
    # Save results
    os.makedirs('results/moving_average_v3', exist_ok=True)
    
    trades_df = pd.DataFrame(portfolio.trades)
    values_df = pd.DataFrame(portfolio.daily_values)
    
    if len(trades_df) > 0:
        trades_df.to_csv('results/moving_average_v3/trade_log.csv', index=False)
        print(f"\nTrade log: results/moving_average_v3/trade_log.csv")
    
    values_df.to_csv('results/moving_average_v3/portfolio_performance.csv', index=False)
    print(f"Performance: results/moving_average_v3/portfolio_performance.csv")
    
    plot_results(metrics, data, TICKERS)
    
    print("\n✅ COMPLETE!")
    print("\nV3 CHANGES:")
    print("✓ Back to 20/50 MA (catches more signals)")
    print("✓ NO STOP-LOSS (let winners run)")
    print("✓ 7 max positions (more diversification)")
    print("✓ Momentum-based entry priority")