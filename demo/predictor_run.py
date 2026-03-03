from predictor import StockPredictor

predictor = StockPredictor('models/')
info = predictor.get_info('AMZN', '2026-02-18')
regime_code, result = predictor.run_prediction(info)

print(regime_code)
print(result)

'''
Output:
2 # HIGH-VOLATILITY
{'ticker': 'AMZN', 'signal': 'BUY', 'sell_pct': np.float64(33.1), 'hold_pct': np.float64(25.6), 'buy_pct': np.float64(41.3), 'confidence': 'High', 'conf_emoji': '🔵', 'signal_no_reg': 'SELL', 'delta_buy': np.float64(7.42), 'error': None}


Understanding Output:
Regime code is 0,1, or 2 (BULLISH, BEARISH, HIGH-VOLATILITY)
in results:
    signal is the prediction with regime
    signal_no_regi is the prediction with no regime
    
    if signal != signal_no_reg, then do not show delta_buy
    else, show the direction of buy probability by delta_buy (+ve => increased direction otherwise decreased direction)
    
    if error is not None: print the error msg (it is either that the ticker is delisted like TWTR is delisted and is now X or the date asked for has no information due to market being closed.) 
    else show:
        ticker
        signal
        signal_no_reg
        sell_pct
        buy_pct
        hold_pct
        confidence
        conf_emoji (if you want)
        delta_buy (depending on the condition above)
        
'''
