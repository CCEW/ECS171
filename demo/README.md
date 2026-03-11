# Stock Predictor Dashboard

A professional Streamlit dashboard for market regime-aware stock classification framework using S&P 500 stocks, featuring multi-stock comparison and normalized trend analysis.

## Features

### Single Stock Prediction
- **Buy/Sell/Hold Signals** - recommendations for any S&P 500 stock on a given date
- **Probability Metrics** - Buy %, Hold %, and Sell % confidence levels
- **Market Regime Detection** - Bullish, Bearish, or High-Volatility classification
- **Visual Signal Indicators** - Color-coded predictions (Green=Buy, Red=Sell, Yellow=Hold)

### Multi-Stock Comparison
- **Normalized Stock Comparison Chart** - Compare up to 6 stocks on the same chart
  - This visualization helps investors identify correlated movements, spot sector trends, and make portfolio rebalancing decisions
- **Dynamic Time Horizon Selector** - Choose from 1 Month to 20 Years
- **Best & Worst Performers** - Market context showing which stocks outperform others

### Professional Dashboard Layout
- **Responsive Grid Layout** - Left panel for prediction controls, right panel for multi-stock comparison
- **Organized Containers** - Bordered sections for easy visual separation

## Dashboard Layout

The dashboard uses a **[1, 2] column responsive grid** with the following structure:

**Left Column (1 unit):**
- Prediction Settings container (ticker, date, predict button)
- Selected Stocks Comparison (best & worst signals)

**Right Column (2 units):**
- Multi-Stock Analysis container (ticker selection, time horizon)
- Normalized Stock Trend Comparison chart

**Below Main Grid (Full Width):**
- Prediction Results section (after "Predict" button clicked):
  - Signal Display (BUY/SELL/HOLD with confidence)
  - Signal Probabilities (Buy %, Hold %, Sell %, Market Regime)
  - Stock Trend Analysis (model buy probability over time)
  - Historical Price Chart (with Bollinger Bands)

## Installation & Usage

### Prerequisites
```bash
Python 3.11
pip
```

### Setup
```bash
cd /workspaces/ECS171/demo
pip install -r ../requirements.txt
```

### Running the Dashboard
```bash
streamlit run demo.py
```

The app will open at `http://localhost:8501`

## How to Use

### Step 1: Single Stock Prediction (Left Panel)
1. Select a stock ticker from the dropdown (e.g., AAPL, MSFT)
2. Choose a prediction date (any past trading date)
3. Click **"Predict"** to generate a recommendation

### Step 2: Multi-Stock Comparison (Right Panel)
1. Select up to 6 stocks from the multiselect box
2. Choose a time horizon (1 Month → 20 Years)
3. Watch the normalized trend chart update automatically

### Understanding the Results
- **Green (BUY)**: Model predicts upward price movement
- **Red (SELL)**: Model predicts downward price movement  
- **Yellow (HOLD)**: Model predicts sideways movement
- **Confidence Levels**: 🔵 High | 🟡 Medium | 🔴 Low

## Model Architecture

### Two-Stage Prediction System

**Stage 1: Market Regime Detection**
- Classifies market conditions as Bullish, Bearish, or High-Volatility
- Trained on 9 technical features

**Stage 2: Signal Generation**
- Generates Buy/Hold/Sell recommendations
- Regime-aware: adjusts signals based on detected market regime
- Provides probability scores for each class

### Technical Features
```
Momentum Indicators:  RSI, MACD, Return lags
Volatility Measures:  ATR, High-Low range, Bollinger Bands
Trend Indicators:     Price vs SMA50, Momentum deviation
Market Structure:     Regime classification
```

## Why See the Graphs?

### Single Stock Chart
The price trend visualization provides visual context by showing:
- **Uptrend/Downtrend patterns** - Validate model signals against price action
- **Support/Resistance levels** - Identify key price zones via Bollinger Bands
- **Volatility assessment** - Understand price stability from band width
- **20-Day SMA** - See intermediate trend direction with Bollinger middle band

### Multi-Stock Comparison
Normalized stock comparison helps you:
- **Identify correlated stocks** - See which holdings move together
- **Spot sector trends** - Compare companies in the same industry
- **Assess relative performance** - Which stocks outperform peers
- **Make rebalancing decisions** - Data-driven portfolio adjustments

## Configuration Files

 `requirements.txt`
lists all Python dependencies:
- `streamlit` - Web framework
- `yfinance` - Stock data API
- `plotly` - Interactive charts
- `altair` - Declarative visualization
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `xgboost`, `randomforest` - ML prediction models

## Troubleshooting

### "No data returned by yfinance"
- Verify ticker is correct and exists in S&P 500
- Check the date is a valid trading day

### "No trading data on [date]"
- Selected date is likely a weekend/holiday
- Try a different date within the trading calendar

### "NaN in computed features"
- Not enough historical data for the stock
- Try selecting a later date or established company

## Model Limitations

- **Historical Only** - Can only predict for dates up to latest market data
- **Delisted Stocks** - Cannot handle stocks removed from S&P 500 (e.g., TWTR → X)
- **Weekend/Holiday** - No predictions for non-trading days
- **New Tickers** - Reduced accuracy for stocks <120 trading days old

## Disclaimer

**Educational Use Only**: Stock predictions are NOT guarantees of future performance. Always consult a financial advisor before making investment decisions. Past performance does not indicate future results.

## Development Notes

### Adding New Features
1. Update helper functions in `demo.py`
2. Test with `streamlit run demo.py`

### Performance Optimization
- First load: ~2-3 seconds (model cached)
- Predictions: ~5-10 seconds (yfinance + feature computation)
- Charts: Instant (Plotly + Altair rendering)

## File Structure

```
demo/
├── demo.py                     # Main Streamlit application
├── predictor.py                # Core ML prediction engine
├── confidence_distribution.py  # Example of confidence distribution
├── predictor_run.py            # Example usage
├── models/                     # Pre-trained model files
│   ├── rf_s1.pkl               # Regime (stage 1)
│   ├── xgb_s2.pkl              # Signal (stage 2)
│   ├── scaler_s1.pkl           # Feature scaler (stage 1)
│   └── scaler_s2.pkl           # Feature scaler (stage 2)
└── README.md                   # This file
```

---

**Last Updated**: March 2026  
**Version**: 3.0 (Multi-stock comparison, analysis, and updates)

