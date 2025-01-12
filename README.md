# Cryptocurrency Dashboard with Technical Analysis and LSTM Predictions

## Overview

This project is a Streamlit-based web application that allows users to visualize cryptocurrency price data, apply various technical analysis indicators, and make future price predictions using an LSTM (Long Short-Term Memory) model.

## Features

- **Cryptocurrency Selection**: Choose from popular cryptocurrencies like Bitcoin (BTC-USD), Ethereum (ETH-USD), Ripple (XRP-USD), Cardano (ADA-USD), and Solana (SOL-USD).
- **Time Range Selection**: Select different time ranges for the data chart (e.g., 1 day, 5 days, 1 month, etc.).
- **Technical Indicators**: Apply multiple technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands.
- **Interactive Charts**: Use Plotly for creating interactive candlestick charts and overlaying technical indicators.
- **LSTM Prediction**: Make future price predictions using an LSTM model based on historical data.

## Prerequisites

Before running the dashboard, ensure you have the following installed:

- Python 3.x
- Required Python packages: `streamlit`, `yfinance`, `pandas`, `plotly`, `ta`, `scikit-learn`, `tensorflow`

You can install the required packages using pip:

```bash
pip install streamlit yfinance pandas plotly ta scikit-learn tensorflow
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
- **Prediction Settings**:
  - **Forecast Days**: Use the slider to select the number of days you want to forecast into the future.
  - **Predict with LSTM**: Click the button to generate predictions using the LSTM model.

### Main Interface

- **Candlestick Chart**: Displays the selected cryptocurrency's price data in a candlestick format.
- **Technical Indicators**: If selected, the chosen technical indicators will be overlaid on the chart.
- **LSTM Forecast Chart**: Displays both historical prices and the predicted future prices using the LSTM model.

## Example Screenshots

![Dashboard Screenshot](path_to_screenshot.png)

*Note: Replace `path_to_screenshot.png` with the actual path to a screenshot of your dashboard.*

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this `README.md` file according to your specific needs, including adding more detailed instructions, screenshots, and any additional features or sections you may have added to the dashboard.

If you need further customization or have any other requirements, let me know! Here’s a more detailed version of the `README.md`:

---

# Cryptocurrency Dashboard with Technical Analysis and LSTM Predictions

## Overview

This project is a Streamlit-based web application designed to provide users with the ability to visualize cryptocurrency price data, apply various technical analysis indicators, and make future price predictions using an LSTM (Long Short-Term Memory) model. The dashboard leverages Plotly for interactive visualizations and the `ta` library for calculating technical indicators.

## Features

- **Cryptocurrency Selection**: Choose from popular cryptocurrencies like Bitcoin (BTC-USD), Ethereum (ETH-USD), Ripple (XRP-USD), Cardano (ADA-USD), and Solana (SOL-USD).
- **Time Range Selection**: Select different time ranges for the data chart (e.g., 1 day, 5 days, 1 month, etc.).
- **Technical Indicators**: Apply multiple technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands.
- **Interactive Charts**: Use Plotly for creating interactive candlestick charts and overlaying technical indicators.
- **LSTM Prediction**: Make future price predictions using an LSTM model based on historical data.

## Prerequisites

Before running the dashboard, ensure you have the following installed:

- Python 3.x
- Required Python packages: `streamlit`, `yfinance`, `pandas`, `plotly`, `ta`, `scikit-learn`, `tensorflow`

You can install the required packages using pip:

```bash
pip install streamlit yfinance pandas plotly ta scikit-learn tensorflow
```

## Running the Dashboard

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ikhwanand/crypto-dashboard.git
   cd crypto-dashboard
   ```

2. **Run the Streamlit App**:

   ```bash
   streamlit run dashboard.py
   ```

3. **Open Your Web Browser**: Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

### Sidebar

- **Select Cryptocurrency**: Choose the cryptocurrency you want to analyze from the dropdown menu.
- **Select Time Range**: Choose the time range for the data chart from the dropdown menu.
- **Select Technical Indicators**: Choose one or more technical indicators to apply to the chart.
- **Prediction Settings**:
  - **Forecast Days**: Use the slider to select the number of days you want to forecast into the future.
  - **Predict with LSTM**: Click the button to generate predictions using the LSTM model.

### Main Interface

- **Candlestick Chart**: Displays the selected cryptocurrency's price data in a candlestick format.
- **Technical Indicators**: If selected, the chosen technical indicators will be overlaid on the chart.
- **LSTM Forecast Chart**: Displays both historical prices and the predicted future prices using the LSTM model.

## Example Screenshots

### Cryptocurrency Price Chart with Technical Indicators

## Additional Notes
If you're using predictions please select the data timestamp you want to predict from the dropdown menu atleast one year.

### Data Source

The data used in this dashboard is fetched from Yahoo Finance using the `yfinance` library.

### Model Details

The LSTM model is a basic sequential model built using Keras. It consists of two LSTM layers followed by dropout layers to prevent overfitting. The model is trained on normalized closing price data with a time step of 60 days.

### Future Improvements

- **Model Tuning**: Hyperparameter tuning and cross-validation can improve the accuracy of predictions.
- **Advanced Models**: Explore more advanced architectures like GRU or Transformer models for better performance.
- **Additional Features**: Incorporate additional features like technical indicators as inputs to the LSTM model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.