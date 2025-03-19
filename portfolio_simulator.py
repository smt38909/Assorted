import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import streamlit as st

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
    """Plots the distribution of ending portfolio values in McKinsey style."""
    plt.style.use('default')  # Reset to default style

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate histogram counts and bins
    counts, bins = np.histogram(ending_values, bins=50)  # Adjust number of bins as needed
    
    # Create a bar plot using the histogram data
    bar_width = (bins[1] - bins[0]) * 0.8  # Reduce width for gaps
    ax.bar(bins[:-1], counts, width=bar_width, color='#00509d', edgecolor='white', linewidth=0.5)  # McKinsey blue
    
    # Customize appearance for McKinsey style
    ax.set_xlabel('Ending Portfolio Value', fontsize=12, fontweight='bold', color='#555555')  # Darker gray
    ax.set_ylabel('Number of Simulations', fontsize=12, fontweight='bold', color='#555555')
    ax.set_title('Distribution of Ending Portfolio Values', fontsize=14, fontweight='bold', color='#333333')  # Darker title

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')  # Light gray axis lines
    ax.spines['bottom'].set_color('#cccccc')

    ax.tick_params(axis='x', colors='#555555', labelsize=10)
    ax.tick_params(axis='y', colors='#555555', labelsize=10)
    ax.yaxis.grid(False)  # Remove y-axis gridlines

    # Add median and percentile lines with more subtle styling
    median_val = np.median(ending_values)
    percentile_10 = np.percentile(ending_values, 10)
    percentile_90 = np.percentile(ending_values, 90)

    ax.axvline(median_val, color='#d9534f', linestyle='-', linewidth=1, label=f'Median: ₹{median_val:,.2f}')  # Red, solid line
    ax.axvline(percentile_10, color='#5bc0de', linestyle='--', linewidth=1, label=f'10th Pctl: ₹{percentile_10:,.2f}')  # Cyan, dashed
    ax.axvline(percentile_90, color='#5bc0de', linestyle='--', linewidth=1, label=f'90th Pctl: ₹{percentile_90:,.2f}')

    ax.legend(fontsize=10, frameon=False)  # Remove legend frame
    plt.tight_layout()
    return fig

# --- Streamlit App ---

st.title("Portfolio Simulator")
st.markdown("Run Monte Carlo simulations to estimate your portfolio's future value.")

# --- Input Section ---
st.sidebar.header("Simulation Parameters")

start_value = st.sidebar.number_input("Starting Portfolio Value (₹)", min_value=1.0, value=1000000.00, step=10000.00, format="%.2f")
start_age = st.sidebar.number_input("Starting Age", min_value=18, max_value=100, value=30, step=1)
target_age = st.sidebar.number_input("Calculate Portfolio Value At Age", min_value=start_age + 1, max_value=120, value=60, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=100000, value=10000, step=1000)

st.sidebar.subheader("Asset Allocation")
stock_percentage = st.sidebar.slider("Percentage in Stocks (%)", 0, 100, 70)
bond_percentage = 100 - stock_percentage
st.sidebar.write(f"Percentage in Bonds: {bond_percentage}%")

st.sidebar.subheader("Expected Return Ranges (%)")
stock_min_return_percent = st.sidebar.text_input("Minimum Expected Stock Return (%)", "5")
stock_max_return_percent = st.sidebar.text_input("Maximum Expected Stock Return (%)", "12")
bond_min_return_percent = st.sidebar.text_input("Minimum Expected Bond Return (%)", "1")
bond_max_return_percent = st.sidebar.text_input("Maximum Expected Bond Return (%)", "5")

num_years = target_age - start_age

if st.button("Run Simulation"):
    if num_years <= 0:
        st.error("Target age must be greater than starting age.")
    try:
        stock_min_return = float(stock_min_return_percent) / 100
        stock_max_return = float(stock_max_return_percent) / 100
        bond_min_return = float(bond_min_return_percent) / 100
        bond_max_return = float(bond_max_return_percent) / 100

        if stock_min_return >= stock_max_return or bond_min_return >= bond_max_return:
            st.error("Minimum expected return must be less than the maximum expected return for both stocks and bonds.")
        else:
            with st.spinner("Running simulations..."):
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

            st.subheader("Simulation Results")

            avg_ending_value = np.mean(ending_portfolio_values)
            median_ending_value = np.median(ending_portfolio_values)
            percentile_10 = np.percentile(ending_portfolio_values, 10)
            percentile_90 = np.percentile(ending_portfolio_values, 90)

            st.markdown(f"**Average Ending Value:** ₹{avg_ending_value:,.2f}")
            st.markdown(f"**Median Ending Value:** ₹{median_ending_value:,.2f}")
            st.markdown(f"**10th Percentile:** ₹{percentile_10:,.2f} (10% of simulations ended below this value)")
            st.markdown(f"**90th Percentile:** ₹{percentile_90:,.2f} (90% of simulations ended below this value)")

            st.subheader("Distribution of Ending Portfolio Values")
            fig = plot_distribution(ending_portfolio_values)
            st.pyplot(fig)

            st.info("Disclaimer: This is a simplified simulation and does not account for all real-world factors such as inflation, taxes, fees, and market volatility. Investment involves risk, and past performance is not indicative of future results.")

    except ValueError:
        st.error("Please enter valid numerical values for the expected return ranges.")
