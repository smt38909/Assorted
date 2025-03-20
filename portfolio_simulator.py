<<<<<<< HEAD
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- Helper Functions ---

def generate_uniform_returns(min_return, max_return, mean_return, num_simulations, num_years):
    """Generates normally distributed returns centered around mean_return and bounded by min/max."""
    # Calculate standard deviation - using 1/4 of the range as before
    std = (max_return - min_return) / 4
    
    # Generate normally distributed returns centered around mean_return
    returns = np.random.normal(mean_return, std, size=(num_simulations, num_years))
    
    # Clip values to ensure they stay within bounds
    returns = np.clip(returns, min_return, max_return)
    
    return returns

def run_portfolio_simulation(start_value, stock_allocation, bond_allocation,
                             stock_min_return, stock_max_return, stock_mean_return,
                             bond_min_return, bond_max_return,
                             num_simulations, num_years,
                             annual_withdrawal, annual_contribution, inflation_rate):
    """Runs Monte Carlo simulations for portfolio growth with random returns, contributions, and inflation-adjusted withdrawals."""
    stock_returns = generate_uniform_returns(stock_min_return, stock_max_return, stock_mean_return, num_simulations, num_years)
    # For bonds, we'll use the middle of min/max as mean
    bond_mean_return = (bond_min_return + bond_max_return) / 2
    bond_returns = generate_uniform_returns(bond_min_return, bond_max_return, bond_mean_return, num_simulations, num_years)

    portfolio_values = np.zeros((num_simulations, num_years + 1))
    portfolio_values[:, 0] = start_value

    for year in range(num_years):
        stock_return = stock_returns[:, year]
        bond_return = bond_returns[:, year]
        portfolio_return = (stock_allocation * stock_return + bond_allocation * bond_return)
        
        # Calculate inflation-adjusted values for this year
        withdrawal = annual_withdrawal * (1 + inflation_rate) ** year
        contribution = annual_contribution * (1 + inflation_rate) ** year
        
        # Apply returns, add contribution, and subtract withdrawal
        portfolio_values[:, year + 1] = (portfolio_values[:, year] * (1 + portfolio_return) 
                                       + contribution - withdrawal)
        
        # Ensure portfolio doesn't go negative
        portfolio_values[:, year + 1] = np.maximum(portfolio_values[:, year + 1], 0)

    return portfolio_values[:, -1]

def plot_distribution(ending_values):
    """Plots the distribution of ending portfolio values in The Economist style."""
    # Set the style to a clean, minimal base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure and axis with specific size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate histogram data
    counts, bins, _ = ax.hist(ending_values, bins=50, density=True, alpha=0)
    
    # Create filled distribution curve
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.fill_between(bin_centers, counts, alpha=0.6, color='#246B9C', label='Distribution')
    ax.plot(bin_centers, counts, color='#246B9C', linewidth=2)
    
    # Calculate and plot key statistics
    median_val = np.median(ending_values)
    percentile_10 = np.percentile(ending_values, 10)
    percentile_90 = np.percentile(ending_values, 90)
    
    # Add vertical lines for statistics
    ax.axvline(median_val, color='#E3120B', linestyle='-', linewidth=2, 
               label=f'Median: ₹{median_val:,.0f}')
    ax.axvline(percentile_10, color='#767676', linestyle='--', linewidth=1.5, 
               label=f'10th percentile: ₹{percentile_10:,.0f}')
    ax.axvline(percentile_90, color='#767676', linestyle='--', linewidth=1.5, 
               label=f'90th percentile: ₹{percentile_90:,.0f}')
    
    # Customize appearance
    ax.set_title('Distribution of Portfolio Values at Target Age', 
                fontsize=16, fontweight='bold', pad=20, 
                fontfamily='sans-serif', color='#2f2f2f')
    
    ax.set_xlabel('Portfolio Value (₹)', fontsize=12, fontfamily='sans-serif', 
                 color='#2f2f2f', labelpad=10)
    ax.set_ylabel('Probability Density', fontsize=12, fontfamily='sans-serif', 
                 color='#2f2f2f', labelpad=10)
    
    # Format x-axis with comma separator for thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    plt.xticks(rotation=45)
    
    # Customize grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#cccccc')
    ax.grid(False, axis='x')
    
    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#767676')
    ax.spines['bottom'].set_color('#767676')
    
    # Customize ticks
    ax.tick_params(axis='both', colors='#767676', labelsize=10)
    
    # Add legend with custom styling
    legend = ax.legend(frameon=True, facecolor='white', framealpha=1, 
                      edgecolor='#cccccc', fontsize=10)
    for text in legend.get_texts():
        text.set_color('#2f2f2f')
    
    # Add explanatory text
    text_y_pos = ax.get_ylim()[1] * 0.95
    ax.text(percentile_10, text_y_pos, '10% chance of\nlower value',
            color='#767676', fontsize=9, ha='right', va='top')
    ax.text(percentile_90, text_y_pos, '10% chance of\nhigher value',
            color='#767676', fontsize=9, ha='left', va='top')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Streamlit UI
st.set_page_config(page_title="Portfolio Simulator", layout="wide")

st.title("Portfolio Growth Simulator")
st.write("""
This tool simulates potential portfolio growth using Monte Carlo simulation. 
Adjust the parameters below to see how different scenarios might affect your investment outcomes.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Portfolio Parameters")
    
    start_value = st.number_input("Starting Portfolio Value (₹)", 
                                 min_value=1.0, 
                                 value=1000000.0, 
                                 step=100000.0,
                                 format="%.2f")
    
    start_age = st.number_input("Current Age", 
                               min_value=18, 
                               max_value=80, 
                               value=30)
    
    target_age = st.number_input("Target Age", 
                                min_value=start_age + 1, 
                                max_value=100, 
                                value=60)
    
    stock_percentage = st.slider("Stock Allocation (%)", 
                               min_value=0, 
                               max_value=100, 
                               value=70)
    
    st.header("Withdrawal Parameters")
    annual_withdrawal = st.number_input("Annual Withdrawal Amount (₹)", 
                                      min_value=0.0, 
                                      value=50000.0, 
                                      step=10000.0,
                                      format="%.2f")
    
    st.header("Contribution Parameters")
    annual_contribution = st.number_input("Annual Contribution Amount (₹)", 
                                      min_value=0.0, 
                                      value=100000.0, 
                                      step=10000.0,
                                      format="%.2f")
    
    inflation_rate = st.slider("Expected Annual Inflation Rate (%)", 
                              min_value=0.0, 
                              max_value=10.0, 
                              value=6.0) / 100.0
    
    st.header("Return Assumptions")
    stock_min_return = st.slider("Minimum Stock Return (%)", 
                                min_value=-25.0, 
                                max_value=25.0, 
                                value=5.0) / 100.0
    
    stock_max_return = st.slider("Maximum Stock Return (%)", 
                                min_value=float(stock_min_return * 100 + 1), 
                                max_value=25.0, 
                                value=12.0) / 100.0
    
    stock_mean_return = st.slider("Average Stock Return (%)",
                                min_value=float(stock_min_return * 100),
                                max_value=float(stock_max_return * 100),
                                value=float((stock_min_return + stock_max_return) * 50)) / 100.0
    
    bond_min_return = st.slider("Minimum Bond Return (%)", 
                                min_value=-5.0, 
                                max_value=10.0, 
                                value=1.0) / 100.0
    
    bond_max_return = st.slider("Maximum Bond Return (%)", 
                                min_value=float(bond_min_return * 100 + 1), 
                                max_value=15.0, 
                                value=5.0) / 100.0
    
    num_simulations = st.slider("Number of Simulations", 
                               min_value=1000, 
                               max_value=20000, 
                               value=10000)

# Calculate parameters
bond_percentage = 100 - stock_percentage
num_years = target_age - start_age

# Run simulation
ending_portfolio_values = run_portfolio_simulation(
    start_value,
    stock_percentage / 100,
    bond_percentage / 100,
    stock_min_return,
    stock_max_return,
    stock_mean_return,
    bond_min_return,
    bond_max_return,
    num_simulations,
    num_years,
    annual_withdrawal,
    annual_contribution,
    inflation_rate
)

# Display results
col1, col2 = st.columns([2, 1])

with col1:
    st.pyplot(plot_distribution(ending_portfolio_values))

with col2:
    st.header("Simulation Results")
    
    avg_ending_value = np.mean(ending_portfolio_values)
    median_ending_value = np.median(ending_portfolio_values)
    percentile_10 = np.percentile(ending_portfolio_values, 10)
    percentile_90 = np.percentile(ending_portfolio_values, 90)
    
    st.metric("Average Ending Value", f"₹{avg_ending_value:,.2f}")
    st.metric("Median Ending Value", f"₹{median_ending_value:,.2f}")
    st.metric("10th Percentile", f"₹{percentile_10:,.2f}")
    st.metric("90th Percentile", f"₹{percentile_90:,.2f}")

st.markdown("""
---
**Disclaimer:** This is a simplified simulation and does not account for all real-world factors
such as inflation, taxes, fees, and market volatility. Investment involves risk, and
past performance is not indicative of future results.
""") 
=======
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- Helper Functions ---

def generate_uniform_returns(min_return, max_return, num_simulations, num_years):
    """Generates uniformly distributed random returns within the given bounds."""
    returns = np.random.uniform(min_return, max_return, size=(num_simulations, num_years))
    return returns

def run_portfolio_simulation(start_value, stock_allocation, bond_allocation,
                             stock_min_return, stock_max_return,
                             bond_min_return, bond_max_return,
                             num_simulations, num_years):
    """Runs Monte Carlo simulations for portfolio growth with uniform random returns."""
    stock_returns = generate_uniform_returns(stock_min_return, stock_max_return, num_simulations, num_years)
    bond_returns = generate_uniform_returns(bond_min_return, bond_max_return, num_simulations, num_years)

    portfolio_values = np.zeros((num_simulations, num_years + 1))
    portfolio_values[:, 0] = start_value

    for year in range(num_years):
        stock_return = stock_returns[:, year]
        bond_return = bond_returns[:, year]
        portfolio_return = (stock_allocation * stock_return + bond_allocation * bond_return)
        portfolio_values[:, year + 1] = portfolio_values[:, year] * (1 + portfolio_return)

    return portfolio_values[:, -1]

def plot_distribution(ending_values):
    """Plots the distribution of ending portfolio values in The Economist style."""
    # Set the style to a clean, minimal base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure and axis with specific size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate histogram data
    counts, bins, _ = ax.hist(ending_values, bins=50, density=True, alpha=0)
    
    # Create filled distribution curve
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.fill_between(bin_centers, counts, alpha=0.6, color='#246B9C', label='Distribution')
    ax.plot(bin_centers, counts, color='#246B9C', linewidth=2)
    
    # Calculate and plot key statistics
    median_val = np.median(ending_values)
    percentile_10 = np.percentile(ending_values, 10)
    percentile_90 = np.percentile(ending_values, 90)
    
    # Add vertical lines for statistics
    ax.axvline(median_val, color='#E3120B', linestyle='-', linewidth=2, 
               label=f'Median: ₹{median_val:,.0f}')
    ax.axvline(percentile_10, color='#767676', linestyle='--', linewidth=1.5, 
               label=f'10th percentile: ₹{percentile_10:,.0f}')
    ax.axvline(percentile_90, color='#767676', linestyle='--', linewidth=1.5, 
               label=f'90th percentile: ₹{percentile_90:,.0f}')
    
    # Customize appearance
    ax.set_title('Distribution of Portfolio Values at Target Age', 
                fontsize=16, fontweight='bold', pad=20, 
                fontfamily='sans-serif', color='#2f2f2f')
    
    ax.set_xlabel('Portfolio Value (₹)', fontsize=12, fontfamily='sans-serif', 
                 color='#2f2f2f', labelpad=10)
    ax.set_ylabel('Probability Density', fontsize=12, fontfamily='sans-serif', 
                 color='#2f2f2f', labelpad=10)
    
    # Format x-axis with comma separator for thousands
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    plt.xticks(rotation=45)
    
    # Customize grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#cccccc')
    ax.grid(False, axis='x')
    
    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#767676')
    ax.spines['bottom'].set_color('#767676')
    
    # Customize ticks
    ax.tick_params(axis='both', colors='#767676', labelsize=10)
    
    # Add legend with custom styling
    legend = ax.legend(frameon=True, facecolor='white', framealpha=1, 
                      edgecolor='#cccccc', fontsize=10)
    for text in legend.get_texts():
        text.set_color('#2f2f2f')
    
    # Add explanatory text
    text_y_pos = ax.get_ylim()[1] * 0.95
    ax.text(percentile_10, text_y_pos, '10% chance of\nlower value',
            color='#767676', fontsize=9, ha='right', va='top')
    ax.text(percentile_90, text_y_pos, '10% chance of\nhigher value',
            color='#767676', fontsize=9, ha='left', va='top')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Streamlit UI
st.set_page_config(page_title="Portfolio Simulator", layout="wide")

st.title("Portfolio Growth Simulator")
st.write("""
This tool simulates potential portfolio growth using Monte Carlo simulation. 
Adjust the parameters below to see how different scenarios might affect your investment outcomes.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Portfolio Parameters")
    
    start_value = st.number_input("Starting Portfolio Value (₹)", 
                                 min_value=1.0, 
                                 value=1000000.0, 
                                 step=100000.0,
                                 format="%.2f")
    
    start_age = st.number_input("Current Age", 
                               min_value=18, 
                               max_value=80, 
                               value=30)
    
    target_age = st.number_input("Target Age", 
                                min_value=start_age + 1, 
                                max_value=100, 
                                value=60)
    
    stock_percentage = st.slider("Stock Allocation (%)", 
                               min_value=0, 
                               max_value=100, 
                               value=70)
    
    st.header("Return Assumptions")
    stock_min_return = st.slider("Minimum Stock Return (%)", 
                                min_value=-25.0, 
                                max_value=25.0, 
                                value=5.0) / 100.0
    
    stock_max_return = st.slider("Maximum Stock Return (%)", 
                                min_value=float(stock_min_return * 100 + 1), 
                                max_value=25.0, 
                                value=12.0) / 100.0
    
    bond_min_return = st.slider("Minimum Bond Return (%)", 
                                min_value=-5.0, 
                                max_value=10.0, 
                                value=1.0) / 100.0
    
    bond_max_return = st.slider("Maximum Bond Return (%)", 
                                min_value=float(bond_min_return * 100 + 1), 
                                max_value=15.0, 
                                value=5.0) / 100.0
    
    num_simulations = st.slider("Number of Simulations", 
                               min_value=1000, 
                               max_value=20000, 
                               value=10000)

# Calculate parameters
bond_percentage = 100 - stock_percentage
num_years = target_age - start_age

# Run simulation
ending_portfolio_values = run_portfolio_simulation(
    start_value,
    stock_percentage / 100,
    bond_percentage / 100,
    stock_min_return,
    stock_max_return,
    bond_min_return,
    bond_max_return,
    num_simulations,
    num_years
)

# Display results
col1, col2 = st.columns([2, 1])

with col1:
    st.pyplot(plot_distribution(ending_portfolio_values))

with col2:
    st.header("Simulation Results")
    
    avg_ending_value = np.mean(ending_portfolio_values)
    median_ending_value = np.median(ending_portfolio_values)
    percentile_10 = np.percentile(ending_portfolio_values, 10)
    percentile_90 = np.percentile(ending_portfolio_values, 90)
    
    st.metric("Average Ending Value", f"₹{avg_ending_value:,.2f}")
    st.metric("Median Ending Value", f"₹{median_ending_value:,.2f}")
    st.metric("10th Percentile", f"₹{percentile_10:,.2f}")
    st.metric("90th Percentile", f"₹{percentile_90:,.2f}")

st.markdown("""
---
**Disclaimer:** This is a simplified simulation and does not account for all real-world factors
such as inflation, taxes, fees, and market volatility. Investment involves risk, and
past performance is not indicative of future results.
""") 
>>>>>>> 21624e7eb4492ee1fe7e7d5d7128565f77dee943
