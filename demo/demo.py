import streamlit as st
import pandas as pd
from datetime import datetime
from predictor import StockPredictor

'''
Implement Streamlit Demo ui here.
Usage of StockPredictor:
    predictor = StockPredictor('models/')
    info = predictor.get_info('NVDA', '2026-02-26') i.e (Ticker, Date)
    regime_code, result = predictor.run_prediction(info)
    
Look at predictor_run once to understand the usage and ui expectations
'''
