# Portfolio Simulator

This is a simple web application built with Streamlit to simulate the potential future value of an investment portfolio based on user-defined parameters. It runs multiple Monte Carlo simulations using a uniform random distribution for asset returns within specified ranges.

## Features

* **User-Friendly Input:** Allows users to easily input their starting portfolio value, age, target age, number of simulations, and asset allocation (stocks/bonds).
* **Flexible Return Assumptions:** Instead of fixed average returns and standard deviations, users can define a minimum and maximum expected return percentage for both stocks and bonds. The simulator then generates random returns within these bounds for each simulation year.
* **Monte Carlo Simulation:** Runs a specified number of simulations to model various potential portfolio growth paths.
* **Key Outcome Statistics:** Displays the average, median, 10th percentile, and 90th percentile of the simulated ending portfolio values. This provides insights into the range of possible outcomes.
* **Distribution Visualization:** Presents a histogram of the ending portfolio values in a style inspired by The Economist, allowing users to visualize the probability distribution of potential outcomes.
* **Web-Based Interface:** Accessible through any web browser thanks to Streamlit.

## How to Use

1.  **Clone or Download the Repository:** If you have the code in a GitHub repository, clone it to your local machine. If you have the Python file directly, save it to a directory.
2.  **Install Dependencies:** Make sure you have the necessary Python libraries installed. Open your terminal or command prompt, navigate to the directory containing the script, and run:
    ```bash
    pip install numpy pandas matplotlib seaborn scipy streamlit
    ```
3.  **Run the Application:** In the same terminal or command prompt, run the Streamlit app:
    ```bash
    streamlit run your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of the Python file, e.g., `portfolio_simulator.py`).
4.  **Interact with the App:** Your web browser will automatically open, displaying the Portfolio Simulator.
    * **Sidebar Inputs:** Use the sidebar on the left to enter your simulation parameters:
        * **Starting Portfolio Value (₹):** The initial value of your investment portfolio.
        * **Starting Age:** Your current age.
        * **Calculate Portfolio Value At Age:** The age at which you want to project the portfolio value.
        * **Number of Simulations:** The number of Monte Carlo simulations to run (more simulations provide a more robust estimate).
        * **Percentage in Stocks (%):** The percentage of your portfolio allocated to stocks. The remaining percentage will be allocated to bonds.
        * **Expected Return Ranges (%):** Enter the minimum and maximum expected annual return percentages for both stocks and bonds.
    * **Run Simulation:** Once you have entered all the parameters, click the "Run Simulation" button.
    * **View Results:** The main area of the app will display the simulation results, including:
        * Average Ending Value
        * Median Ending Value
        * 10th Percentile Ending Value (10% of simulations ended below this)
        * 90th Percentile Ending Value (90% of simulations ended below this)
        * A distribution graph visualizing the range of ending portfolio values.
    * **Run Again:** You can modify the input parameters in the sidebar and click "Run Simulation" again to perform a new simulation.

## Assumptions

* **Uniform Return Distribution:** The simulator assumes that the annual returns for stocks and bonds follow a uniform random distribution within the minimum and maximum ranges you provide. This is a simplification of real-world market behavior.
* **Constant Asset Allocation:** The portfolio's asset allocation (percentage in stocks and bonds) remains constant throughout the simulation period.
* **No Additional Contributions or Withdrawals:** The simulation does not account for any additional investments or withdrawals made after the initial starting value.
* **No Fees or Taxes:** The calculations do not include any investment fees, transaction costs, or taxes.
* **Annual Returns:** Returns are considered on an annual basis.

## Disclaimer

This Portfolio Simulator is a tool for illustrative purposes only and should not be considered financial advice. The results are based on random simulations and the assumptions you provide, which may not accurately reflect future market conditions. Investment involves risk, and you could lose money. Consult with a qualified financial advisor for personalized financial planning.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit pull requests with your enhancements or bug fixes.

## License

Created by AK@2025
