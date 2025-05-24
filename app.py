import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
import threading

# Global variables to track training progress & stop flag
training_progress = 0
is_training = False
stop_training = False

# ------------------------------
# Sidebar Design
# ------------------------------
# st.sidebar.image("i4.jpg", width=200)
st.sidebar.title("📊 Stock Price Prediction")
st.sidebar.markdown("An interactive app for Stock Prediction and Analysis:")
page = st.sidebar.radio("📌 Navigate through the sections below:", ["Home", "Dataset Visualization", "Trained Model Visualization", "Train Model", "Evaluation", "Predict"])

st.sidebar.markdown("---")  # Divider

# ------------------------------
# Common Functions
# ------------------------------
@st.cache_data
def load_stock_data(stock, start, end):
    try:
        data = yf.download(stock, start=start, end=end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def save_model_and_scaler(model, scaler, stock):
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    st.success(f"Model and scaler saved as {model_path} and {scaler_path}.")

def train_model_thread(stock, start_date, end_date, model_filename, scaler_filename):
    global training_progress, is_training, stop_training
    is_training = True
    training_progress = 0

    from train_lstm import train_lstm  # Import the training function

    try:
        model, scaler = train_lstm(stock, start_date, end_date, model_filename)

        # Simulate training progress
        for percent in range(101):
            if stop_training:
                st.warning("⏹️ Training Stopped by User.")
                is_training = False
                return
            training_progress = percent
            time.sleep(0.1)  # Simulating training time

        save_model_and_scaler(model, scaler, model_filename, scaler_filename)
        st.success(f"🎉 Model training complete for {stock}!")
        is_training = False

    except Exception as e:
        st.error(f"⚠️ Error during training: {e}")
        is_training = False


def fetch_and_save_stock_data(stock, file_path):
    """Fetch latest stock data and save as CSV."""
    df = yf.download(stock, period="1y", interval="1d")
    df.reset_index(inplace=True)
    df.to_csv(file_path, index=False)
    return df

def get_latest_stock_values(df):
    """Extract latest close price, MA100, and MA200."""
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Convert, set non-numeric to NaN
    df = df.dropna()  # Remove any rows with NaN values after conversion
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    if df['Close'].dtype != 'float64' and df['Close'].dtype != 'int64':
        st.error("Close column is not numeric. Please check the dataset.")
    else:
        df['MA100'] = df['Close'].rolling(window=100).mean()
    latest_close = df['Close'].iloc[-1]
    latest_ma100 = df['MA100'].iloc[-1]
    latest_ma200 = df['MA200'].iloc[-1]
    return latest_close, latest_ma100, latest_ma200

# ------------------------------
# Home Page
# ------------------------------
if page == "Home":
    st.title("📈 Stock Price Prediction System")
    st.markdown("This tool allows you to visualize and predict stock prices using **LSTM models**.")

    st.info("👉 **How to Use:** Select a stock and date range, then explore the data.")
    st.divider()

    # User Input for Stock Selection and Date Range
    stock = st.selectbox("📍 Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"], index=0)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("📆 Start Date", pd.to_datetime('2012-01-01'))
    end_date = col2.date_input("📆 End Date", pd.to_datetime('2022-12-21'))

    data = load_stock_data(stock, start_date, end_date)
    if data is not None:
        st.success(f"✅ Loaded data for {stock}")
        st.dataframe(data.style.set_properties(**{'background-color': 'black', 'color': 'white'}))

        # Download Button
        st.download_button("📥 Download Data", data.to_csv(index=False), file_name=f"{stock}_data.csv", mime="text/csv")

elif page == "Dataset Visualization":
    st.title("📊 Dataset Visualization")

    # User Input for Stock Selection and Date Range
    stock = st.selectbox("📍 Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"], index=0)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("📆 Start Date", pd.to_datetime('2012-01-01'))
    end_date = col2.date_input("📆 End Date", pd.to_datetime('2022-12-21'))

    # Load Data
    data = load_stock_data(stock, start_date, end_date)

    if data is not None:
        st.success(f"✅ Loaded stock data for {stock}")

        # Debugging: Print DataFrame and columns
        st.write("### Raw Data Preview")
        st.dataframe(data.head())

        st.write("### DataFrame Columns")
        st.write(data.columns.tolist())

        # Ensure 'Date' column is in datetime format
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
        else:
            st.error("❌ 'Date' column not found in the data.")
            st.stop()

        # Moving Averages Calculation
        data["MA_50"] = data["Close"].rolling(window=50).mean()
        data["MA_100"] = data["Close"].rolling(window=100).mean()
        data["MA_200"] = data["Close"].rolling(window=200).mean()

        # Drop rows with NaN values (due to rolling averages)
        data.dropna(inplace=True)

        # Debugging: Print DataFrame after cleaning
        st.write("### Cleaned Data Preview")
        st.dataframe(data.head())

        # Debugging: Check data types
        st.write("### Data Types")
        st.write(data.dtypes)

        # Tabbed Visualization
        tab1, tab2, tab3 = st.tabs(["📌 Price vs MA50", "📌 Price vs MA50 & MA100", "📌 Price vs MA100 & MA200"])

        with tab1:
            st.write(f"#### {stock} - Price vs MA50")
            try:
                fig1 = px.line(data, x="Date", y=["Close", "MA_50"], title=f"{stock} - Price vs MA50")
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating plot: {e}")

        with tab2:
            st.write(f"#### {stock} - Price vs MA50 & MA100")
            try:
                fig2 = px.line(data, x="Date", y=["Close", "MA_50", "MA_100"], title=f"{stock} - Price vs MA50 & MA100")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating plot: {e}")

        with tab3:
            st.write(f"#### {stock} - Price vs MA100 & MA200")
            try:
                fig3 = px.line(data, x="Date", y=["Close", "MA_100", "MA_200"], title=f"{stock} - Price vs MA100 & MA200")
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating plot: {e}")

        # Download Data with Moving Averages
        st.download_button(
            label="📥 Download Data with Moving Averages",
            data=data.to_csv(index=False),
            file_name=f"{stock}_data_with_ma.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Failed to load stock data. Please check your inputs and try again.")

# ------------------------------
# Train Model Page
# ------------------------------

elif page == "Train Model":
    st.title("🚀 Train a New Model")

    # User Input for Stock Selection and Date Range
    st.subheader("📌 Select Stock and Date Range")
    stock = st.selectbox("📍 Choose Stock", ["GOOG", "AAPL", "AMZN", "MSFT"], index=0)

    col1, col2 = st.columns(2)
    start_date = col1.date_input("📆 Start Date", pd.to_datetime('2012-01-01'))
    end_date = col2.date_input("📆 End Date", pd.to_datetime('2022-12-21'))

    # Dynamic file naming based on selection
    model_filename = f"{stock}_LSTM_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.keras"
    scaler_filename = f"{stock}_Scaler_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.save"

    # If training is in progress, show progress
    if is_training:
        st.warning(f"⏳ Model training for {stock} is in progress... {training_progress}% completed.")
        if st.button("⏹️ Stop Training"):
            stop_training = True
            st.warning("Stopping the training process...")

    else:
        if st.button("🚀 Start Training", key="start_training", help="Click to start training"):
            stop_training = False
            training_thread = threading.Thread(target=train_model_thread, args=(stock, start_date, end_date, model_filename, scaler_filename))
            training_thread.start()
            st.rerun()

    # Display progress bar if training is running
    if is_training:
        progress_bar = st.progress(training_progress / 100)

    # Display Model Summary if training is done
    if not is_training and training_progress == 100:
        st.write("### 📊 Model Summary")
        with st.expander("🔍 Click to Expand Model Summary"):
            model_summary = model.summary()
            st.table(model_summary)

        # Download Buttons for Model & Scaler
        st.write("### 📥 Download Trained Files")
        col1, col2 = st.columns(2)

        with open(model_filename, "rb") as f:
            col1.download_button(
                label="📥 Download Model",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream"
            )

        with open(scaler_filename, "rb") as f:
            col2.download_button(
                label="📥 Download Scaler",
                data=f,
                file_name=scaler_filename,
                mime="application/octet-stream"
            )

elif page == "Evaluation":
    st.title("🔍 Model Evaluation")
    st.markdown("Evaluate the performance of the trained LSTM model on the selected stock data.")

    # User Input for Stock Selection and Date Range
    stock = st.selectbox("📍 Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"], index=0)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("📆 Start Date", pd.to_datetime('2012-01-01'))
    end_date = col2.date_input("📆 End Date", pd.to_datetime('2022-12-21'))

    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("✅ Loaded pretrained model and scaler.")

        # Load Data
        data = load_stock_data(stock, start_date, end_date)
        if data is not None:
            # Feature Engineering
            data['MA_100'] = data['Close'].rolling(100).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data.dropna(inplace=True)

            features = ['Close', 'MA_100', 'MA_200']
            test_scaled = scaler.transform(data[features])

            # Create Sequences
            def create_sequences(data, time_steps=100):
                x, y = [], []
                for i in range(time_steps, len(data)):
                    x.append(data[i - time_steps:i])
                    y.append(data[i, 0])  # Predicting 'Close' price
                return np.array(x), np.array(y)

            x_test, y_test = create_sequences(test_scaled)

            # Model Prediction
            with st.spinner("Making predictions... This may take a few seconds."):
                y_pred_scaled = model.predict(x_test)
                y_pred = y_pred_scaled * (1 / scaler.scale_[0])  # Inverse transform

            # Ensure y_test and y_pred are 1-dimensional
            y_test = y_test.flatten()
            y_pred = y_pred.flatten()

            # Evaluation Metrics
            st.subheader("📊 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")

            # Visualize Actual vs Predicted Prices
            st.subheader("📈 Actual vs Predicted Prices")
            fig = px.line(title=f"Actual vs Predicted Prices for {stock}")
            fig.add_scatter(x=data['Date'].iloc[100:], y=y_test, mode='lines', name='Actual Price', line=dict(color='blue'))
            fig.add_scatter(x=data['Date'].iloc[100:], y=y_pred, mode='lines', name='Predicted Price', line=dict(color='red'))
            st.plotly_chart(fig, use_container_width=True)

            # Download Evaluation Results
            evaluation_results = pd.DataFrame({
                "Date": data['Date'].iloc[100:].values,  # Ensure Date is 1-dimensional
                "Actual Price": y_test,
                "Predicted Price": y_pred
            })
            st.download_button(
                label="📥 Download Evaluation Results",
                data=evaluation_results.to_csv(index=False),
                file_name=f"{stock}_evaluation_results.csv",
                mime="text/csv"
            )

            # Model Summary (Expandable Section)
            with st.expander("📝 Model Summary"):
                st.write("### Model Architecture")
                model.summary(print_fn=lambda x: st.text(x))

    else:
        st.warning("⚠️ No pretrained model found. Please train a model first.")

# ------------------------------
# Predict Page
# ------------------------------
# 🔮 PREDICTION SECTION
elif page == "Predict":
    st.title("🔮 Predict New Values")

    # Stock Selection
    stock = st.selectbox("📍 Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"], index=0)
    stock_symbol = {"GOOG": "GOOGL", "AAPL": "AAPL", "AMZN": "AMZN", "MSFT": "MSFT"}[stock]
    stock_data_path = f"{stock}_historical_data.csv"

    # Fetch latest stock data if file doesn't exist
    if not os.path.exists(stock_data_path):
        st.info("Fetching latest stock data...")
        df = fetch_and_save_stock_data(stock_symbol, stock_data_path)
    else:
        df = pd.read_csv(stock_data_path)
        df['Date'] = pd.to_datetime(df['Date'])

    latest_close, latest_ma100, latest_ma200 = get_latest_stock_values(df)

    # Load model and scaler
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Loaded pretrained model and scaler.")

        # User Input
        st.write("### Enter Input Values for Prediction")
        close = st.slider("Close Price", min_value=latest_close * 0.95, max_value=latest_close * 1.05, value=latest_close)
        ma_100 = st.slider("100-Day Moving Average", min_value=latest_ma100 * 0.95, max_value=latest_ma100 * 1.05, value=latest_ma100)
        ma_200 = st.slider("200-Day Moving Average", min_value=latest_ma200 * 0.95, max_value=latest_ma200 * 1.05, value=latest_ma200)

        if st.button("Predict"):
            input_data = np.array([[close, ma_100, ma_200]])
            input_scaled = scaler.transform(input_data)
            input_scaled = input_scaled.reshape((1, 1, 3))
            prediction_scaled = model.predict(input_scaled)
            prediction = prediction_scaled * (1 / scaler.scale_[0])
            st.success(f"Predicted Close Price: {prediction[0][0]:.2f}")

            # Visualization
            st.write("### Prediction Visualization")
            fig = px.line(title=f"Predicted Close Price for {stock}")
            fig.add_scatter(x=[0], y=[prediction[0][0]], mode='markers', name='Predicted Price', marker=dict(color="red", size=10))
            st.plotly_chart(fig)
    else:
        st.warning("No pretrained model found. Please train a model first.")
