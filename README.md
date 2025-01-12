# Cryptocurrency Dashboard with Technical Analysis

## Overview

This project is a Streamlit-based web application that allows users to visualize cryptocurrency price data and apply various technical analysis indicators. The dashboard uses Plotly for interactive charting and the `ta` library for calculating technical indicators.

## Features

- **Cryptocurrency Selection**: Choose from popular cryptocurrencies like Bitcoin (BTC-USD), Ethereum (ETH-USD), Ripple (XRP-USD), Cardano (ADA-USD), and Solana (SOL-USD).
- **Time Range Selection**: Select different time ranges for the data chart (e.g., 1 day, 5 days, 1 month, etc.).
- **Technical Indicators**: Apply multiple technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands.
- **Interactive Charts**: Use Plotly for creating interactive candlestick charts and overlaying technical indicators.

## Prerequisites

Before running the dashboard, ensure you have the following installed:

- Python 3.x
- Required Python packages: `streamlit`, `yfinance`, `pandas`, `plotly`, `ta`

You can install the required packages using pip:

```bash
pip install streamlit yfinance pandas plotly ta
```

## Running the Dashboard

1. Clone the repository:

   ```bash
   git clone https://github.com/ikhwanand/crypto-dashboard.git
   cd crypto-dashboard
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run dashboard.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

### Sidebar

- **Select Cryptocurrency**: Choose the cryptocurrency you want to analyze from the dropdown menu.
- **Select Time Range**: Choose the time range for the data chart from the dropdown menu.
- **Select Technical Indicators**: Choose one or more technical indicators to apply to the chart.

### Main Interface

- **Candlestick Chart**: Displays the selected cryptocurrency's price data in a candlestick format.
- **Technical Indicators**: If selected, the chosen technical indicators will be overlaid on the chart.
- **Interactive Controls**: Use Plotly's interactive controls to zoom in/out, pan, and hover over data points for detailed information.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

