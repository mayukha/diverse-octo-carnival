import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load data
print("Loading data...")
combined = pd.read_csv('data/combined_3y.csv', index_col=0, parse_dates=True)
print(f"Data loaded: {combined.shape[0]} days, {combined.shape[1]} stocks")
print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}\n")

# Create results/eda folder
import os
os.makedirs('results/eda', exist_ok=True)

# ============================================
# 1. NORMALIZED PRICE MOVEMENTS (All stocks)
# ============================================
print("Creating normalized price chart...")
fig, ax = plt.subplots(figsize=(15, 8))

normalized = combined / combined.iloc[0] * 100

for col in normalized.columns:
    ax.plot(normalized.index, normalized[col], label=col, linewidth=2)

ax.set_title('Normalized Price Movements (Base = 100)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Normalized Price', fontsize=12)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/eda/1_normalized_prices.png', dpi=300)
print("✅ Saved: results/eda/1_normalized_prices.png")

# ============================================
# 2. INDIVIDUAL STOCK PERFORMANCE
# ============================================
print("Creating individual stock charts...")
fig, axes = plt.subplots(5, 2, figsize=(15, 18))
axes = axes.flatten()

for idx, col in enumerate(combined.columns):
    ax = axes[idx]
    prices = combined[col].dropna()
    
    # Calculate return
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    ax.plot(prices.index, prices, linewidth=2, color='#2E86AB')
    ax.fill_between(prices.index, prices, alpha=0.3, color='#2E86AB')
    ax.set_title(f'{col} | Return: {total_return:+.1f}%', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))

plt.tight_layout()
plt.savefig('results/eda/2_individual_stocks.png', dpi=300)
print("✅ Saved: results/eda/2_individual_stocks.png")

# ============================================
# 3. RETURNS & VOLATILITY
# ============================================
print("Calculating returns and volatility...")

# Calculate daily returns
returns = combined.pct_change().dropna()

# Calculate metrics
metrics = pd.DataFrame({
    'Total_Return_%': ((combined.iloc[-1] / combined.iloc[0]) - 1) * 100,
    'Volatility_%': returns.std() * np.sqrt(252) * 100,
    'Sharpe_Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
    'Max_Drawdown_%': [(combined[col] / combined[col].cummax() - 1).min() * 100 
                       for col in combined.columns]
})

metrics = metrics.sort_values('Total_Return_%', ascending=False)
print("\n" + "="*60)
print("PERFORMANCE METRICS (3 Years):")
print("="*60)
print(metrics.to_string())
print("="*60)

# Save metrics
metrics.to_csv('results/eda/performance_metrics.csv')
print("\n✅ Saved: results/eda/performance_metrics.csv")

# Plot returns vs volatility
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(metrics['Volatility_%'], metrics['Total_Return_%'], 
                     s=200, alpha=0.6, c=range(len(metrics)), cmap='viridis')

for idx, row in metrics.iterrows():
    ax.annotate(idx, (row['Volatility_%'], row['Total_Return_%']), 
                fontsize=11, fontweight='bold')

ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
ax.set_ylabel('3-Year Total Return (%)', fontsize=12)
ax.set_title('Risk-Return Profile', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/eda/3_risk_return.png', dpi=300)
print("✅ Saved: results/eda/3_risk_return.png")

# ============================================
# 4. CORRELATION MATRIX
# ============================================
print("Creating correlation matrix...")
corr_matrix = returns.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix (Daily Returns)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/eda/4_correlation_matrix.png', dpi=300)
print("✅ Saved: results/eda/4_correlation_matrix.png")

# ============================================
# 5. DRAWDOWN ANALYSIS
# ============================================
print("Creating drawdown chart...")
fig, ax = plt.subplots(figsize=(15, 8))

for col in combined.columns:
    cummax = combined[col].cummax()
    drawdown = (combined[col] / cummax - 1) * 100
    ax.plot(drawdown.index, drawdown, label=col, linewidth=1.5, alpha=0.7)

ax.set_title('Drawdown Analysis (% from Peak)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.legend(loc='lower left', ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('results/eda/5_drawdowns.png', dpi=300)
print("✅ Saved: results/eda/5_drawdowns.png")

print("\n" + "="*60)
print("✅ EDA COMPLETE! Check results/eda/ folder for visualizations")
print("="*60)