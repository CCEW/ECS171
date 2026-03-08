import numpy as np
import pandas as pd
import yfinance as yf
from collections import Counter
from predictor import StockPredictor, S1_FEATURES, S2_FEATURES_BASE

import warnings
warnings.filterwarnings('ignore')


def f(S, E):
    TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA', 'IBM']

    predictor = StockPredictor('models')
    confidence_counts = Counter()

    for ticker in TICKERS:
        print(f'Processing {ticker}...')

        raw = yf.download(ticker, start=S, end=E, auto_adjust=True, progress=False, threads=False)

        if raw.empty:
            print(f'  No data for {ticker}, skipping.')
            continue

        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        feat_df = predictor._compute_features(raw)

        all_features = list(dict.fromkeys(S1_FEATURES + S2_FEATURES_BASE))
        valid_rows = feat_df[all_features].dropna()

        for date, row in valid_rows.iterrows():
            s1_vals = row[S1_FEATURES].values.flatten()
            s2_base_vals = row[S2_FEATURES_BASE].values.flatten()

            try:
                X_s1 = predictor.scaler_s1.transform(s1_vals.reshape(1, -1))
                regime_code = int(predictor.rf_s1.predict(X_s1)[0])
            except Exception:
                regime_code = 0

            regime_direction = {0: 1.0, 1: -1.0, 2: 0.0}[regime_code]
            regime_vol_flag  = 1.0 if regime_code == 2 else 0.0

            ret_5d  = float(s2_base_vals[S2_FEATURES_BASE.index('Ret_5d')])
            rsi = float(s2_base_vals[S2_FEATURES_BASE.index('RSI')])
            atr_pct = float(s2_base_vals[S2_FEATURES_BASE.index('ATR_Pct')])

            interactions = np.array([
                float(regime_code),
                regime_direction * ret_5d,
                regime_direction * (rsi / 100.0),
                regime_vol_flag  * atr_pct,
            ], dtype=np.float32)

            X_s2 = np.concatenate([s2_base_vals, interactions]).reshape(1, -1)
            X_s2_sc = predictor.scaler_s2.transform(X_s2)
            proba   = predictor.xgb_s2.predict_proba(X_s2_sc)[0]

            sorted_p = np.sort(proba)[::-1]
            margin = sorted_p[0] - sorted_p[1]

            if   margin >= 0.08: confidence = 'High'
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
    print(f'Low:    {confidence_counts["Low"]:>6}  ({100*confidence_counts["Low"]/total:.1f}%)')
    print(f'{"="*40}')


f('2000-01-01', '2026-03-01')
f('2015-01-01', '2025-01-01')  
