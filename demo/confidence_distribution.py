import numpy as np
import pandas as pd
import yfinance as yf
from collections import Counter
from predictor import StockPredictor, TICKER_ENCODING, S1_FEATURES


def f(S, E):
    TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA', 'IBM']

    predictor = StockPredictor('models')
    confidence_counts = Counter()
    error_count = 0

    for ticker in TICKERS:
        print(f'Processing {ticker}...')

        raw = yf.download(ticker, start=S, end=E, auto_adjust=True, progress=False, threads=False)

        if raw.empty:
            print(f'  No data for {ticker}, skipping.')
            continue

        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        feat_df = predictor._compute_features(raw)
        encoded_ticker = TICKER_ENCODING.get(ticker, 250)
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        valid_rows = feat_df[S1_FEATURES].dropna()

        for date, row in valid_rows.iterrows():
            feat_vals = row.values.flatten()

            try:
                X_s1 = predictor.scaler_s1.transform(feat_vals.reshape(1, -1))
                regime_code = int(predictor.lgb_s1.predict(X_s1)[0])
            except Exception:
                regime_code = 0

            X_s2 = np.append(feat_vals, [encoded_ticker, regime_code]).reshape(1, -1)
            X_s2_sc = predictor.scaler_s2.transform(X_s2)
            proba = predictor.lgb_s2.predict_proba(X_s2_sc)[0]

            sorted_p = np.sort(proba)[::-1]
            margin = sorted_p[0] - sorted_p[1]

            if margin >= 0.08:   confidence = 'High'
            elif margin >= 0.04: confidence = 'Medium'
            else: confidence = 'Low'

            confidence_counts[confidence] += 1

        print(f'  Done. Running total: {sum(confidence_counts.values())} predictions.')

    total = sum(confidence_counts.values())
    print()
    print(f'{"="*40}')
    print(f'Total valid predictions: {total}')
    print(f'High:   {confidence_counts["High"]:>6}  ({100*confidence_counts["High"]/total:.1f}%)')
    print(f'Medium: {confidence_counts["Medium"]:>6}  ({100*confidence_counts["Medium"]/total:.1f}%)')
    print(f'Low: {confidence_counts["Low"]:>6}  ({100*confidence_counts["Low"]/total:.1f}%)')
    print(f'{"="*40}')


f('2000-01-01','2026-03-01' ) # for 21st century
f('2015-01-01','2025-01-01' ) # seen data
