'''
DO NOT CHANGE FILE CONTENTS

Usage:
    predictor = StockPredictor('models/')
    info = predictor.get_info('NVDA', '2026-02-26') i.e (Ticker, Date)
    regime_code, result = predictor.run_prediction(info)
'''

import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

warnings.filterwarnings('ignore')

S1_FEATURES = [
    'Ret_Lag1',
    'Ret_5d',
    'RSI',
    'ATR_Pct',
    'Volatility_Ratio',
    'Return_Dispersion',
]

S2_FEATURES_BASE = [
    'Ret_Lag1',
    'Ret_5d',
    'RSI',
    'ATR_Pct',
    'Volatility_Ratio',
    'Price_vs_SMA50',
    'Momentum_Deviation',
    'BB_Pct',
    'Vol_Price_Trend',
]  
INTERACTION_FEATURES = [
    'regime_pred',
    'Dir_x_Ret5',
    'Dir_x_RSI',
    'Vol_x_ATR',
]

S2_FEATURES_REGIME = S2_FEATURES_BASE + INTERACTION_FEATURES

class StockPredictor:
    """
    Two-stage regime-aware stock signal predictor.

    Stage 1: RandomForest (rf_s1.pkl)  → Bullish / Bearish / High-Volatility
    Stage 2: XGBoost    (xgb_s2.pkl)  → Buy / Hold / Sell

    Usage:
        predictor = StockPredictor('models/')
        info = predictor.get_info('NVDA', '2026-02-26')
        regime_code, result = predictor.run_prediction(info)
    """

    SIGNAL_MAP  = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    REGIME_MAP  = {0: 'BULLISH', 1: 'BEARISH', 2: 'HIGH-VOLATILITY'}
    WARMUP_DAYS = 120

    def __init__(self, model_dir: str = 'models/'):
        """Load rf_s1, xgb_s2, scaler_s1, scaler_s2 from model_dir."""
        def _load(fname):
            with open(f'{model_dir}/{fname}', 'rb') as f:
                return pickle.load(f)

        self.rf_s1 = _load('rf_s1.pkl')
        self.xgb_s2   = _load('xgb_s2.pkl')
        self.scaler_s1 = _load('scaler_s1.pkl')
        self.scaler_s2 = _load('scaler_s2.pkl')

    @staticmethod
    def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
        """yfinance sometimes returns multi-level columns — flatten them."""
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    @staticmethod
    def _compute_features(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features needed for S1 and S2 from OHLC+Volume data.
        Requires 60+ rows of history for accurate rolling windows.

        S1 features (6):  Ret_Lag1, Ret_5d, RSI, ATR_Pct,
                          Volatility_Ratio, Return_Dispersion
        S2 base features (9): above (minus Return_Dispersion) +
                          Price_vs_SMA50, Momentum_Deviation, BB_Pct,
                          Vol_Price_Trend
        """
        df = df_raw.copy()
        close  = df['Close']
        high   = df['High']
        low    = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)

        df['Ret_Lag1'] = close.pct_change(1)
        df['Ret_5d']   = close.pct_change(5)

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['ATR_Pct'] = tr.rolling(14).mean() / close

        ret = close.pct_change()
        vol_short = ret.rolling(5).std()
        vol_long  = ret.rolling(20).std()
        df['Volatility_Ratio'] = vol_short / vol_long.replace(0, np.nan)

        df['Return_Dispersion'] = ret.rolling(10).std()

        df['Price_vs_SMA50']    = close / close.rolling(50).mean()

        sma20 = close.rolling(20).mean()
        df['Momentum_Deviation'] = (close - sma20) / sma20

        bb_std = close.rolling(20).std()
        bb_up  = sma20 + 2 * bb_std
        bb_low = sma20 - 2 * bb_std
        df['BB_Pct'] = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)

        vol_norm = volume / volume.rolling(20).mean().replace(0, np.nan)
        df['Vol_Price_Trend'] = vol_norm * df['Ret_Lag1']

        return df

    def get_info(self, ticker: str, target_date_str: str) -> dict:
        """
        Fetches OHLC data from yfinance, computes features, and returns
        everything needed to call run_prediction.

        Parameters:
            ticker e.g. 'NVDA'
            target_date_str e.g. '2026-02-26'

        Returns dict with keys:
            ticker str
            s1_vals np.ndarray  shape (6,)  — S1_FEATURES values
            s2_base_vals    np.ndarray  shape (9,)  — S2_FEATURES_BASE values
            error str | None
        """
        ticker = ticker.strip().upper()
        target_date = pd.to_datetime(target_date_str)
        start_date  = target_date - timedelta(days=self.WARMUP_DAYS)
        end_date    = target_date + timedelta(days=1)

        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return {
                'ticker': ticker, 's1_vals': None, 's2_base_vals': None,
                'error': f'No data returned by yfinance for {ticker}'
            }

        raw = self._flatten_yf_columns(raw)
        feat = self._compute_features(raw)
        target_row = feat[feat.index.date == target_date.date()]

        if target_row.empty:
            return {
                'ticker': ticker, 's1_vals': None, 's2_base_vals': None,
                'error': f'No trading data on {target_date_str} — market may be closed'
            }

        s1_vals = target_row[S1_FEATURES].values.flatten()
        s2_base_vals = target_row[S2_FEATURES_BASE].values.flatten()

        if np.any(np.isnan(s1_vals)) or np.any(np.isnan(s2_base_vals)):
            return {
                'ticker': ticker, 's1_vals': None, 's2_base_vals': None,
                'error': 'NaN in computed features — insufficient history for rolling windows'
            }

        return {
            'ticker': ticker,
            's1_vals': s1_vals,
            's2_base_vals': s2_base_vals,
            'error': None,
        }

    def run_prediction(self, info: dict) -> tuple:
        """
        Runs Stage 1 (regime) then Stage 2 (Buy/Hold/Sell) on get_info output.

        Stage 1: rf_s1 predicts regime from 6 scaled S1_FEATURES.
        Stage 2: interaction features are computed from regime_code, then
                 xgb_s2 predicts signal from 13 scaled S2_FEATURES_REGIME.

        Parameters:
            info    dict returned by get_info

        Returns:
            regime_code  int   0=Bullish | 1=Bearish | 2=High-Volatility
            result dict  {
                ticker, signal, regime, sell_pct, hold_pct, buy_pct,
                confidence, conf_emoji, signal_no_reg, delta_buy, error
            }
        """
        if info['error']: return 0, {'ticker': info['ticker'], 'error': info['error']}

        s1_vals = info['s1_vals']
        s2_base_vals = info['s2_base_vals']

        regime_code = 0
        try:
            X_s1 = self.scaler_s1.transform(s1_vals.reshape(1, -1))
            regime_code = int(self.rf_s1.predict(X_s1)[0])
        except Exception:
            pass

        regime_direction = {0: 1.0, 1: -1.0, 2: 0.0}[regime_code]
        regime_vol_flag  = 1.0 if regime_code == 2 else 0.0

        ret_5d  = float(s2_base_vals[S2_FEATURES_BASE.index('Ret_5d')])
        rsi = float(s2_base_vals[S2_FEATURES_BASE.index('RSI')])
        atr_pct = float(s2_base_vals[S2_FEATURES_BASE.index('ATR_Pct')])

        dir_x_ret5 = regime_direction * ret_5d
        dir_x_rsi  = regime_direction * (rsi / 100.0)
        vol_x_atr  = regime_vol_flag  * atr_pct

        interactions = np.array([
            float(regime_code),
            dir_x_ret5,
            dir_x_rsi,
            vol_x_atr,
        ], dtype=np.float32)

        X_s2_regime = np.concatenate([s2_base_vals, interactions]).reshape(1, -1)
        X_s2_regime_sc = self.scaler_s2.transform(X_s2_regime)
        proba_regime   = self.xgb_s2.predict_proba(X_s2_regime_sc)[0]

        no_regime_interactions = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        X_s2_base = np.concatenate([s2_base_vals, no_regime_interactions]).reshape(1, -1)
        X_s2_base_sc = self.scaler_s2.transform(X_s2_base)
        proba_base   = self.xgb_s2.predict_proba(X_s2_base_sc)[0]

        signal = self.SIGNAL_MAP[int(np.argmax(proba_regime))]
        signal_base = self.SIGNAL_MAP[int(np.argmax(proba_base))]

        sorted_proba = np.sort(proba_regime)[::-1]
        margin = sorted_proba[0] - sorted_proba[1]

        if margin >= 0.08: confidence, conf_emoji = 'High',   '🔵'
        elif margin >= 0.04: confidence, conf_emoji = 'Medium', '🟡'
        else: confidence, conf_emoji = 'Low', '🔴'

        result = {
            'ticker': info['ticker'],
            'signal': signal,
            'regime': self.REGIME_MAP[regime_code],
            'sell_pct': round(proba_regime[0] * 100, 1),
            'hold_pct': round(proba_regime[1] * 100, 1),
            'buy_pct': round(proba_regime[2] * 100, 1),
            'confidence': confidence,
            'conf_emoji':   conf_emoji,
            'signal_no_reg': signal_base,
            'delta_buy':    round((proba_regime[2] - proba_base[2]) * 100, 2),
            'error': None,
        }

        return regime_code, result
