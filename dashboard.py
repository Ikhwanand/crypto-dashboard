import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set page configuration
st.set_page_config(page_title="Crypto Dashboard", layout="wide", page_icon=':chart_with_upwards_trend:')

# Title
st.title("Cryptocurrency Dashboard with Technical Analysis and LSTM Predictions")

# Sidebar for user input
st.sidebar.header("User Inputs")

# Cryptocurrency selection
crypto_symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Ripple (XRP-USD)": "XRP-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Polygon (MATIC-USD)": "MATIC-USD",
    "Toncoin (TON-USD)": "TON-USD",
} # you can add it here coin what you like

selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_symbols.keys()))

# Time range selection
time_ranges = {
    # "1 Day": "1d",
    # "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "YTD": "ytd",
    "Max": "max"
}


# Technical Indicators Selection
indicators = st.sidebar.multiselect("Select Technical Indicators", 
                                    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"])


selected_time_range = st.sidebar.selectbox("Select Time Range", list(time_ranges.keys()))

# Fetch crypto data based on selected time range
@st.cache_data
def fetch_data(symbol, period):
    data = yf.download(symbol, period=period)
    return data

data = fetch_data(crypto_symbols[selected_crypto], time_ranges[selected_time_range])

if not data.empty:
    # Drop any rows with NaN values
    data = dropna(data)
    
    # Ensure the index is a datetime index
    data.index = pd.to_datetime(data.index)

    # Ensure the columns are named correctly
    if 'Adj Close' in data.columns:
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    else:
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    

    # Add technical indicators if selected
    if indicators:
        try:
            data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
        except Exception as e:
            st.error(f"Error adding technical indicators: {e}")
            st.stop()

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Price')])

    # Add moving averages if SMA is selected
    if "SMA" in indicators:
        sma_50 = data['Close'].rolling(window=50).mean()
        sma_100 = data['Close'].rolling(window=100).mean()
        sma_200 = data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=data.index, y=sma_50, mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=data.index, y=sma_100, mode='lines', name='SMA 100'))
        fig.add_trace(go.Scatter(x=data.index, y=sma_200, mode='lines', name='SMA 200'))

    # Add EMA if EMA is selected
    if "EMA" in indicators:
        ema_50 = data['trend_ema_fast']
        fig.add_trace(go.Scatter(x=data.index, y=ema_50, mode='lines', name='EMA 50'))

    # Add RSI if RSI is selected
    if "RSI" in indicators:
        fig = go.Figure()
        
        # Create subplot with shared x-axis
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{selected_crypto} Price Chart", "RSI"))

        # Candlestick chart in the first row
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Price'), row=1, col=1)

        # Add moving averages if SMA is selected
        if "SMA" in indicators:
            sma_50 = data['Close'].rolling(window=50).mean()
            sma_100 = data['Close'].rolling(window=100).mean()
            sma_200 = data['Close'].rolling(window=200).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma_50, mode='lines', name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=sma_100, mode='lines', name='SMA 100'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=sma_200, mode='lines', name='SMA 200'), row=1, col=1)

        # Add EMA if EMA is selected
        if "EMA" in indicators:
            ema_50 = data['trend_ema_fast']
            fig.add_trace(go.Scatter(x=data.index, y=ema_50, mode='lines', name='EMA 50'), row=1, col=1)

        # Add RSI in the second row
        rsi = data['momentum_rsi']
        fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
        
        # Add a horizontal line at 70 and 30 for RSI overbought/oversold levels
        fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data.index), mode='lines', 
                                 name='Overbought (70)', line=dict(color='red', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data.index), mode='lines', 
                                 name='Oversold (30)', line=dict(color='green', dash='dot')), row=2, col=1)

        # Update layout for better visualization
        # fig.update_layout(
        #     title=f"{selected_crypto} Price and RSI Chart",
        #     xaxis_rangeslider_visible=False,
        #     height=800  # Increase height to accommodate subplots
        # )
        
        # # Update y-axis titles
        # fig.update_yaxes(title_text="Price", row=1, col=1)
        # fig.update_yaxes(title_text="RSI", row=2, col=1)


    # Add MACD if MACD is selected
    if "MACD" in indicators:
        macd_line = data['trend_macd']
        signal_line = data['trend_macd_signal']
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode='lines', name='MACD Line'))
        fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode='lines', name='Signal Line'))

    # Add Bollinger Bands if Bollinger Bands is selected
    if "Bollinger Bands" in indicators:
        bb_upper = data['volatility_bbh']
        bb_lower = data['volatility_bbl']
        fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
        fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))

    # Update layout for better visualization
    fig.update_layout(
        title=f"{selected_crypto} Price Chart",
        xaxis_rangeslider_visible=False,
        yaxis=dict(title='Price'),
        xaxis=dict(title='Date')
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)
    
    st.sidebar.subheader("Prediction Settings")
    forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=60, value=7)
    
    if st.sidebar.button("Predict with LSTM"):
        # Prepare data for LSTM
        df = data[['Close']].copy()
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        # Create the training dataset
        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[0:int(training_data_len), :]
        
         # Split into x_train and y_train datasets
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                X.append(a)
                y.append(dataset[i + time_step, 0])
            return np.array(X), np.array(y)
        
        time_step = 60
        x_train, y_train = create_dataset(train_data, time_step)
        x_test, y_test = create_dataset(scaled_data[training_data_len:], time_step)
        
        # Reshape input to be [samples, time steps, features] which is required for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        
        # Create test dataset
        test_data = scaled_data[training_data_len - time_step:, :]
        x_test, y_test = create_dataset(test_data, time_step)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Predict future prices
        predictions = []
        last_sequence = scaled_data[-time_step:]
        for _ in range(forecast_days):
            last_sequence = np.reshape(last_sequence, (1, time_step, 1))
            next_price = model.predict(last_sequence)
            predictions.append(next_price[0, 0])
            last_sequence = np.concatenate((last_sequence[0, 1:, :], next_price), axis=0)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create a DataFrame for the forecasted prices
        forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_days + 1, freq='D')[1:]
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': predictions.flatten()
        })
        
        # Plot the forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
        fig_forecast.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast'))
        fig_forecast.update_layout(
            title=f"{selected_crypto} Price Forecast",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price')
        )
        
        st.plotly_chart(fig_forecast)
        st.write(forecast_df)

else:
    st.write("No data available for the selected time range.")