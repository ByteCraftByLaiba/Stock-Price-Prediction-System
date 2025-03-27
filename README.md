# Stock Price Prediction System

## Description
This is an interactive **Stock Price Prediction System** built using **Streamlit**, **LSTM (Long Short-Term Memory) neural networks**, and **Yahoo Finance API**. It enables users to:
- Visualize historical stock prices with moving averages.
- Train a custom LSTM model for stock price prediction.
- Evaluate the trained model using various performance metrics.
- Predict future stock prices.

## Features
- 📈 **Stock Data Visualization:** Fetch and visualize stock data from Yahoo Finance.
- 🚀 **Train LSTM Model:** Train a custom LSTM model on selected stock data.
- 📊 **Performance Evaluation:** Evaluate the model using RMSE, MAE, and R-squared scores.
- 🔮 **Stock Price Prediction:** Predict future prices based on trained models.
- 🛠 **Download Model & Scaler:** Save and reuse trained models for future predictions.

## Installation
### Prerequisites
Ensure you have Python 3.7+ installed and the required dependencies:
```bash
pip install streamlit pandas yfinance plotly numpy keras scikit-learn joblib
```

### Clone the Repository
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### Run the Application
```bash
streamlit run app.py
```

## Usage
1. **Select a Stock:** Choose from available stocks (GOOG, AAPL, AMZN, MSFT) or enter a custom ticker.
2. **Set Date Range:** Select the start and end dates for analysis.
3. **Visualize Data:** View stock trends and moving averages.
4. **Train the Model:** Train an LSTM model on historical stock data.
5. **Evaluate Performance:** Analyze model accuracy with various metrics.
6. **Make Predictions:** Use the trained model to forecast future stock prices.

## Folder Structure
```
stock-price-prediction/
│── app.py                # Main Streamlit application
│── train_lstm.py         # LSTM model training script
│── models/               # Saved trained models
│── data/                 # Processed stock data
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation
```

## Contributing
Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License.
