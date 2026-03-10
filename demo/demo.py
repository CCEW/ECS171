import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import altair as alt
from datetime import datetime, timedelta
from predictor import StockPredictor,  S1_FEATURES, S2_FEATURES_BASE
import warnings
import os
warnings.filterwarnings('ignore')

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# INITIALIZE SESSION STATE & CACHE

@st.cache_resource
def load_predictor():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    return StockPredictor(model_dir)

predictor = load_predictor()
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA"]
TICKER_ENCODING = {
    'A':1,'AAL':2,'AAP':3,'AAPL':4,'ABBV':5,'ABC':6,'ABMD':7,'ABT':8,
    'ACN':9,'ADBE':10,'ADI':11,'ADM':12,'ADP':13,'ADSK':14,'AEE':15,
    'AEP':16,'AES':17,'AFL':18,'AIG':19,'AIZ':20,'AJG':21,'AKAM':22,
    'ALB':23,'ALGN':24,'ALK':25,'ALL':26,'ALLE':27,'AMAT':28,'AMCR':29,
    'AMD':30,'AME':31,'AMGN':32,'AMP':33,'AMT':34,'AMZN':35,'ANET':36,
    'ANF':37,'AON':38,'AOS':39,'APA':40,'APD':41,'APH':42,'APTV':43,
    'ARE':44,'ATO':45,'AVB':46,'AVGO':47,'AVY':48,'AWK':49,'AXP':50,
    'AZO':51,'BA':52,'BAC':53,'BAX':54,'BBWI':55,'BBY':56,'BDX':57,
    'BEN':58,'BF.B':59,'BIIB':60,'BIO':61,'BK':62,'BKNG':63,'BKR':64,
    'BLK':65,'BMY':66,'BR':67,'BRK.B':68,'BSX':69,'BWA':70,'BXP':71,
    'C':72,'CAG':73,'CAH':74,'CARR':75,'CAT':76,'CB':77,'CBOE':78,
    'CBRE':79,'CCI':80,'CCL':81,'CDAY':82,'CDNS':83,'CDW':84,'CE':85,
    'CEG':86,'CF':87,'CFG':88,'CHD':89,'CHRW':90,'CHTR':91,'CI':92,
    'CINF':93,'CL':94,'CLX':95,'CMA':96,'CMCSA':97,'CME':98,'CMG':99,
    'CMI':100,'CMS':101,'CNC':102,'CNP':103,'COF':104,'COO':105,'COP':106,
    'COST':107,'CPB':108,'CPRT':109,'CPT':110,'CRL':111,'CRM':112,
    'CSCO':113,'CSGP':114,'CSX':115,'CTAS':116,'CTLT':117,'CTRA':118,
    'CTSH':119,'CTVA':120,'CVS':121,'CVX':122,'CZR':123,'D':124,
    'DAL':125,'DD':126,'DE':127,'DFS':128,'DG':129,'DGX':130,'DHI':131,
    'DHR':132,'DIS':133,'DISH':134,'DLR':135,'DLTR':136,'DOV':137,
    'DOW':138,'DPZ':139,'DRI':140,'DTE':141,'DUK':142,'DVA':143,
    'DVN':144,'DXC':145,'DXCM':146,'EA':147,'EBAY':148,'ECL':149,
    'ED':150,'EFX':151,'EG':152,'EIX':153,'EL':154,'EMN':155,'EMR':156,
    'ENPH':157,'EOG':158,'EPAM':159,'EQIX':160,'EQR':161,'EQT':162,
    'ES':163,'ESS':164,'ETN':165,'ETR':166,'ETSY':167,'EVRG':168,
    'EW':169,'EXC':170,'EXPD':171,'EXPE':172,'EXR':173,'F':174,
    'FANG':175,'FAST':176,'FCX':177,'FDS':178,'FDX':179,'FE':180,
    'FFIV':181,'FIS':182,'FISV':183,'FITB':184,'FLT':185,'FMC':186,
    'FOX':187,'FOXA':188,'FRC':189,'FRT':190,'FTNT':191,'FTV':192,
    'GD':193,'GE':194,'GILD':195,'GIS':196,'GL':197,'GLW':198,
    'GM':199,'GNRC':200,'GOOG':201,'GOOGL':202,'GPC':203,'GPN':204,
    'GRMN':205,'GS':206,'GWW':207,'HAL':208,'HAS':209,'HBAN':210,
    'HCA':211,'HD':212,'HES':213,'HIG':214,'HII':215,'HLT':216,
    'HOLX':217,'HON':218,'HPE':219,'HPQ':220,'HRL':221,'HSIC':222,
    'HST':223,'HSY':224,'HUM':225,'HWM':226,'IBM':227,'ICE':228,
    'IDXX':229,'IEX':230,'IFF':231,'ILMN':232,'INCY':233,'INTC':234,
    'INTU':235,'IP':236,'IPG':237,'IQV':238,'IR':239,'IRM':240,
    'ISRG':241,'IT':242,'ITW':243,'IVZ':244,'J':245,'JBHT':246,
    'JCI':247,'JKHY':248,'JNJ':249,'JNPR':250,'JPM':251,'K':252,
    'KDP':253,'KEY':254,'KEYS':255,'KHC':256,'KIM':257,'KLAC':258,
    'KMB':259,'KMI':260,'KMX':261,'KO':262,'KR':263,'L':264,
    'LDOS':265,'LEN':266,'LH':267,'LHX':268,'LIN':269,'LKQ':270,
    'LLY':271,'LMT':272,'LNC':273,'LNT':274,'LOW':275,'LRCX':276,
    'LUMN':277,'LUV':278,'LVS':279,'LW':280,'LYB':281,'LYV':282,
    'MA':283,'MAA':284,'MAR':285,'MAS':286,'MCD':287,'MCHP':288,
    'MCK':289,'MCO':290,'MDLZ':291,'MDT':292,'MET':293,'META':294,
    'MGM':295,'MHK':296,'MKC':297,'MKTX':298,'MLM':299,'MMC':300,
    'MMM':301,'MNST':302,'MO':303,'MOH':304,'MOS':305,'MPC':306,
    'MPWR':307,'MRK':308,'MRNA':309,'MRO':310,'MS':311,'MSCI':312,
    'MSFT':313,'MSI':314,'MTB':315,'MTCH':316,'MTD':317,'MU':318,
    'NCLH':319,'NDAQ':320,'NDSN':321,'NEE':322,'NEM':323,'NFLX':324,
    'NI':325,'NKE':326,'NOC':327,'NOW':328,'NRG':329,'NSC':330,
    'NTAP':331,'NTRS':332,'NUE':333,'NVDA':334,'NVR':335,'NWL':336,
    'NWS':337,'NWSA':338,'NXPI':339,'O':340,'ODFL':341,'OGN':342,
    'OKE':343,'OMC':344,'ON':345,'ORCL':346,'ORLY':347,'OTIS':348,
    'OXY':349,'PARA':350,'PAYC':351,'PAYX':352,'PCAR':353,'PCG':354,
    'PEAK':355,'PEG':356,'PEP':357,'PFE':358,'PFG':359,'PG':360,
    'PGR':361,'PH':362,'PHM':363,'PKG':364,'PLD':365,'PM':366,
    'PNC':367,'PNR':368,'PNW':369,'POOL':370,'PPG':371,'PPL':372,
    'PRU':373,'PSA':374,'PSX':375,'PTC':376,'PWR':377,'PXD':378,
    'PYPL':379,'QCOM':380,'QRVO':381,'RCL':382,'RE':383,'REG':384,
    'REGN':385,'RF':386,'RJF':387,'RL':388,'RMD':389,'ROK':390,
    'ROL':391,'ROP':392,'ROST':393,'RSG':394,'RTX':395,'SBAC':396,
    'SBUX':397,'SCHW':398,'SEDG':399,'SHW':400,'SIVB':401,'SJM':402,
    'SLB':403,'SNA':404,'SNPS':405,'SO':406,'SPG':407,'SPGI':408,
    'SRE':409,'STE':410,'STT':411,'STX':412,'STZ':413,'SWK':414,
    'SWKS':415,'SYF':416,'SYK':417,'SYY':418,'T':419,'TAP':420,
    'TDG':421,'TDY':422,'TECH':423,'TEL':424,'TER':425,'TFC':426,
    'TFX':427,'TGT':428,'TJX':429,'TMO':430,'TMUS':431,'TPR':432,
    'TRMB':433,'TROW':434,'TRV':435,'TSCO':436,'TSLA':437,'TSN':438,
    'TT':439,'TTWO':440,'TXN':441,'TXT':442,'TYL':443,'UAL':444,
    'UDR':445,'UHS':446,'ULTA':447,'UNH':448,'UNP':449,'UPS':450,
    'URI':451,'USB':452,'V':453,'VFC':454,'VLO':455,'VMC':456,
    'VNO':457,'VRSN':458,'VRTX':459,'VTR':460,'VTRS':461,'VZ':462,
    'WAB':463,'WAT':464,'WBA':465,'WBD':466,'WDC':467,'WEC':468,
    'WELL':469,'WFC':470,'WHR':471,'WM':472,'WMB':473,'WMT':474,
    'WRB':475,'WRK':476,'WST':477,'WY':478,'WYNN':479,'XEL':480,
    'XOM':481,'XRAY':482,'XYL':483,'YUM':484,'ZBH':485,'ZBRA':486,
    'ZION':487,'ZTS':488,
}

def stocks_to_str(stocks):
    return ",".join(stocks)


if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None
if 'last_date' not in st.session_state:
    st.session_state.last_date = None
if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")
if 'selected_horizon' not in st.session_state:
    st.session_state.selected_horizon = '6 Months'

# Get all S&P 500 tickers
STOCKS = sorted(TICKER_ENCODING.keys())

# Time horizon mapping
HORIZON_MAP = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 1825,
    "10 Years": 3650,
    "20 Years": 7300,
}

# HELPER FUNCTIONS

def get_signal_color(signal):
    """Map signal to color for visual identification."""
    if signal == 'BUY':
        return '#00d97e'
    elif signal == 'SELL':
        return '#ff6b6b'
    else:  # HOLD
        return '#ffd700'

def get_signal_emoji(signal):
    """Get emoji for signal."""
    if signal == 'BUY':
        return '📈'
    elif signal == 'SELL':
        return '📉'
    else:
        return '⏸️'

def get_price_performance(ticker, end_date, days=90):
    """
    Calculate percentage return over the past days.
    Returns: (current_price, percent_change)
    """
    try:
        hist = get_stock_data_cached(ticker, max_days=3650)
        if hist is None or hist.empty or len(hist) < 2:
            return None, None
        
        # Slice to the last 'days' of data
        hist = hist[hist.index >= pd.Timestamp(end_date - timedelta(days=days))]
        if len(hist) < 2:
            return None, None
        
        close = hist['Close']
        current_price = close.iloc[-1]
        prev_price = close.iloc[0]
        pct_change = ((current_price - prev_price) / prev_price) * 100
        return current_price, pct_change
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_normalized_stock_data(tickers, end_date, days=180):
    """
    Fetch and normalize stock price data for comparison.
    Cached for 1 hour to improve performance.
    
    Normalized prices allow comparison of stocks with different price levels,
    making it easy to see which stocks are trending up or down together.
    This helps investors spot correlations and market trends across multiple holdings.
    """
    normalized_data = {}
    
    for ticker in tickers:
        try:
            hist = get_stock_data_cached(ticker, max_days=3650)
            
            if hist is None or hist.empty or len(hist) < 2:
                continue
            
            # Slice to the last 'days' of data
            hist = hist[hist.index >= pd.Timestamp(end_date - timedelta(days=days))]
            if len(hist) < 2:
                continue
            
            close = hist['Close']
            normalized = (close - close.iloc[0]) / close.iloc[0]
            normalized_data[ticker] = normalized
        except:
            continue
    
    return pd.DataFrame(normalized_data) if normalized_data else None

def plot_normalized_stocks_chart(tickers, end_date, days=180):
    """
    Plot normalized stock prices for comparison.
    
    This comparison chart helps investors:
    • Identify which stocks are outperforming others in your portfolio
    • Spot sector trends by comparing related stocks
    • Understand volatility differences between holdings
    • Make rebalancing decisions based on relative performance
    """
    df = get_normalized_stock_data(tickers, end_date, days)
    
    if df is None or df.empty:
        return None
    
    df = df.reset_index()
    df.columns.name = None
    
    # Melt for Altair
    melted = df.melt(id_vars=['Date'], var_name='Stock', value_name='Normalized Price')
    
    chart = alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Normalized Price:Q', title='Normalized Price (%)'),
        color=alt.Color('Stock:N', title='Stock'),
        tooltip=['Date:T', 'Stock:N', alt.Tooltip('Normalized Price:Q', format='.2%')]
    ).properties(
        height=400,
        width=800
    ).interactive()
    
    return chart

def plot_stock_chart(ticker, end_date, days=180):
    """
    Plot stock price trend with technical indicators.
    
    This visualization helps users understand the historical price movement
    and trends leading up to the prediction date. Seeing the actual price
    trajectory provides confidence in the model's recommendation by showing
    momentum, support/resistance levels, and recent volatility patterns.
    The chart context helps investors make informed decisions beyond just
    the buy/sell/hold signal.
    """
    try:
        hist = get_stock_data_cached(ticker, max_days=3650)
        
        if hist is None or hist.empty:
            st.warning(f"No data available for {ticker}")
            return None
        
        # Slice to the last 'days' of data
        hist = hist[hist.index >= pd.Timestamp(end_date - timedelta(days=days))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#0084ff', width=2),
        ))
        
        # Bollinger Bands (20-day SMA with 2 std dev)
        bb_sma = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)
        
        # Upper band
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=bb_upper,
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
            hoverinfo='skip'
        ))
        
        # Lower band (fill between)
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=bb_lower,
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.1)',
            hoverinfo='skip'
        ))
        
        # Middle band (SMA)
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=bb_sma,
            mode='lines',
            name='20-Day SMA',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'{ticker} - Historical Price Trend (Last {days} Days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_dark',
            height=450,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='rgba(15, 20, 25, 0.5)',
            paper_bgcolor='rgba(15, 20, 25, 0)',
            font=dict(color='#e0e0e0', size=11)
        )
        
        fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.05)', showgrid=True)
        fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.05)', showgrid=True)
        
        return fig
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_best_worst_from_tickers(tickers_list, date_str):
    """
    Get best and worst performers from a specific list of tickers.
    Cached for 1 hour. Uses get_features_cached to avoid redundant yfinance downloads.
    """
    results = []
    target_date = pd.to_datetime(date_str)
    _all_feats = list(dict.fromkeys(S1_FEATURES + S2_FEATURES_BASE))

    for ticker in tickers_list:
        try:
            feat_df = get_features_cached(ticker, target_date)
            if feat_df is None:
                continue
            target_row = feat_df[feat_df.index.date == target_date.date()]
            if target_row.empty:
                continue
            row = target_row.iloc[0]
            if row[_all_feats].isna().any():
                continue

            s1_vals      = row[S1_FEATURES].values.flatten()
            s2_base_vals = row[S2_FEATURES_BASE].values.flatten()

            info = {'ticker': ticker, 's1_vals': s1_vals,
                    's2_base_vals': s2_base_vals, 'error': None}
            regime_code, result = predictor.run_prediction(info)

            if result.get('error') is None:
                _, pct_change = get_price_performance(ticker, target_date, days=90)
                if pct_change is not None:
                    results.append({
                        'ticker':     ticker,
                        'signal':     result['signal'],
                        'buy_pct':    result['buy_pct'],
                        'performance': pct_change,
                        'confidence': result['confidence']
                    })
        except:
            pass

    if not results:
        return None, None

    df    = pd.DataFrame(results)
    best  = df.nlargest(3, 'buy_pct')
    worst = df.nsmallest(3, 'buy_pct')
    return best, worst

@st.cache_data(ttl=3600)
def get_stock_data_cached(ticker, max_days=3650):
    """
    SINGLE SOURCE for all yfinance downloads.
    Downloads max_days of history once and caches it.
    All functions slice this data as needed - no duplicate downloads.
    Cached for 1 hour.
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=max_days)
        hist = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False)
        if hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist
    except:
        return None

@st.cache_data(ttl=3600)
def get_features_cached(ticker, end_date):
    """Cache feature computation separately for each ticker/date combo."""
    hist = get_stock_data_cached(ticker, max_days=3650)  # Get up to 10 years for features
    if hist is None or len(hist) < 100:
        return None
    feat_df = predictor._compute_features(hist)
    return feat_df

@st.cache_data(ttl=3600)
def predict_past_buy_probability(ticker, end_date, days_ahead=300):
    """
    Get historical model buy probability predictions for trend analysis.
    Limits results to the specified days_ahead timeframe.
    Cached for 1 hour. Fully vectorized — all rows processed in one batch
    per model call instead of a per-row loop.
    """
    try:
        feat_df = get_features_cached(ticker, end_date)
        if feat_df is None:
            return None

        # Filter to only the requested time period
        cutoff_date = pd.Timestamp(end_date - timedelta(days=days_ahead))
        feat_df = feat_df[feat_df.index >= cutoff_date]
        
        if feat_df.empty or len(feat_df) < 2:
            return None

        # Sample to reduce data points if needed (roughly 100 points max for chart)
        sample_step = max(1, len(feat_df) // 100)
        sampled = feat_df.iloc[::sample_step]

        _all_feats = list(dict.fromkeys(S1_FEATURES + S2_FEATURES_BASE))
        sampled = sampled.dropna(subset=_all_feats)
        if sampled.empty:
            return None

        s1_batch = sampled[S1_FEATURES].values        # (N, 6)
        s2_batch = sampled[S2_FEATURES_BASE].values   # (N, 9)

        # Stage 1: one transform + one predict for all rows
        X_s1_sc      = predictor.scaler_s1.transform(s1_batch)
        regime_codes = predictor.rf_s1.predict(X_s1_sc).astype(int)  # (N,)

        # Interaction features — vectorized
        direction_map    = np.array([1.0, -1.0, 0.0])
        regime_dirs      = direction_map[regime_codes]
        regime_vol_flags = (regime_codes == 2).astype(float)

        ret5_idx = S2_FEATURES_BASE.index('Ret_5d')
        rsi_idx  = S2_FEATURES_BASE.index('RSI')
        atr_idx  = S2_FEATURES_BASE.index('ATR_Pct')

        interactions = np.column_stack([
            regime_codes.astype(float),
            regime_dirs  * s2_batch[:, ret5_idx],
            regime_dirs  * (s2_batch[:, rsi_idx] / 100.0),
            regime_vol_flags * s2_batch[:, atr_idx],
        ])  # (N, 4)

        # Stage 2: one transform + one predict_proba for all rows
        X_s2_sc = predictor.scaler_s2.transform(
            np.concatenate([s2_batch, interactions], axis=1)
        )
        probas   = predictor.xgb_s2.predict_proba(X_s2_sc)  # (N, 3)
        buy_pcts = probas[:, 2] * 100

        return {
            'past_dates':       list(sampled.index),
            'past_predictions': list(buy_pcts),
            'prediction_date':  end_date,
        }

    except Exception:
        return None

def plot_predicted_trend_chart(tickers, end_date, days=180):
    """
    Plot historical model trend for stock comparison.
    
    Shows model's historical BUY probability trends:
    - Uses historical model predictions as trend indicator (>50% = bullish, <50% = bearish)
    - Helps investors understand model behavior on past data
    """
    all_data = []
    
    for ticker in tickers:
        pred_data = predict_past_buy_probability(ticker, end_date, days_ahead=days)
        if pred_data:
            # Add historical predictions only
            for date, pred in zip(pred_data['past_dates'], pred_data['past_predictions']):
                all_data.append({
                    'Date': date,
                    'Stock': ticker,
                    'Trend': pred
                })
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    
    # Create Altair chart with historical data only
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Trend:Q', title='BUY Probability (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Stock:N', title='Stock'),
        tooltip=['Date:T', 'Stock:N', alt.Tooltip('Trend:Q', format='.1f')]
    ).properties(
        height=400,
        width=700
    ).interactive()
    
    # Add a baseline at 50%
    baseline = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(color='gray', opacity=0.3).encode(y='y:Q')
    
    return chart + baseline


# Web app header

st.title(" Stock Prediction Dashboard")
st.markdown("Predictions for S&P 500 Stocks")


# MAIN LAYOUT - [1, 2] COLUMN GRID

cols = st.columns([1, 2], gap="medium")

# LEFT COLUMN - Input Controls
with cols[0]:
    top_left_cell = st.container(border=True, vertical_alignment="top")
    
    with top_left_cell:
        st.markdown("##### Prediction Settings")
        
        # Ticker selectbox for single stock prediction
        ticker = st.selectbox(
            "Select Stock Ticker",
            options=STOCKS,
            index=STOCKS.index('AAPL') if 'AAPL' in STOCKS else 0,
            placeholder="Choose a stock",
            key="ticker_input"
        )
        
        # Date input for prediction
        prediction_date = st.date_input(
            "Select Prediction Date",
            value=datetime.now().date() - timedelta(days=1),
            key="date_input"
        )
        
        predict_button = st.button("Predict", width='stretch', type="primary")

# RIGHT COLUMN LAYOUT - Subgrid
with cols[1]:
    right_cols = st.columns(1)
    
    # Comparison Settings Container
    with right_cols[0]:
        comp_container = st.container(border=True)
        
        with comp_container:
            st.markdown("##### Multi-Stock Analysis")
            
            comparison_tickers = st.multiselect(
                "Select stocks to compare",
                options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
                default=st.session_state.tickers_input,
                placeholder="Choose stocks for comparison",
                max_selections=6,
                key="comparison_tickers"
            )
            
            horizon = st.radio(
                "Time Horizon",
                options=list(HORIZON_MAP.keys()),
                index=list(HORIZON_MAP.keys()).index('6 Months'),
                horizontal=True,
                key="horizon"
            )
            
            st.session_state.tickers_input = comparison_tickers
            st.session_state.selected_horizon = horizon

# RIGHT COLUMN - Trend Chart
with cols[1]:
    if comparison_tickers:
        trend_container = st.container(border=True)
        
        with trend_container:
            
            days = HORIZON_MAP.get(horizon, 180)
            chart = plot_predicted_trend_chart(comparison_tickers, datetime.now().date(), days=days)
            if chart:
                st.markdown("#### Predicted Trend Comparison", text_alignment="center")
                st.altair_chart(chart, width='stretch')
            else:
                st.info("No prediction data available for selected stocks.")
    else:
        st.info("Select stocks above to see trend analysis.")

# LEFT COLUMN - Best/Worst from Selected Tickers
with cols[0]:
    if comparison_tickers:
        date_str = prediction_date.strftime('%Y-%m-%d')
        best, worst = get_best_worst_from_tickers(comparison_tickers, date_str)
        
        if best is not None and worst is not None:
            perf_container = st.container(border=True)
            
            with perf_container:
                st.markdown("##### Selected Stocks Comparison (BUY%)")
                pcol1, pcol2 = st.columns(2)
                
                with pcol1:
                    st.markdown("**Best Signals**")
                    for idx, row in best.iterrows():
                        st.metric(
                            f"{row['ticker']}",
                            f"{row['buy_pct']:.1f}%",
                            delta = row['confidence'],
                            delta_arrow = "up" if row['confidence'] == 'High' else "down" if row['confidence'] == 'Low' else "off",
                            delta_color= "green" if row['confidence'] == 'High' else "yellow" if row['confidence'] == 'Medium' else "red"
                        )
                
                with pcol2:
                    st.markdown("**Weakest Signals**")
                    for idx, row in worst.iterrows():
                        st.metric(
                            f"{row['ticker']}",
                            f"{row['buy_pct']:.1f}% ",
                            delta= row['confidence'],
                            delta_arrow = "up" if row['confidence'] == 'High' else "down" if row['confidence'] == 'Low' else "off",
                            delta_color= "green" if row['confidence'] == 'High' else "yellow" if row['confidence'] == 'Medium' else "red"
                        )

# PREDICTION RESULTS - Widget Grid Layout


if predict_button or (st.session_state.last_ticker == ticker and st.session_state.last_date == prediction_date and st.session_state.last_ticker is not None):
    st.session_state.last_ticker = ticker
    st.session_state.last_date = prediction_date
    
    date_str = prediction_date.strftime('%Y-%m-%d')
    
    with st.spinner('🔄 Analyzing stock and generating predictions...'):
        info = predictor.get_info(ticker, date_str)
        regime_code, result = predictor.run_prediction(info)
        
        if result.get('error'):
            st.error(f" Error: {result['error']}")
        else:
            # Main Signal Display
            signal = result['signal']
            signal_color = get_signal_color(signal)
            signal_emoji = get_signal_emoji(signal)
            
            signal_container = st.container(border=True)
            with signal_container:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <p style='color: #a0a0a0; margin: 0;'>{ticker} on {date_str}</p>
                    <p style='font-size: 48px; color: {signal_color}; margin: 15px 0; font-weight: bold;'>{signal_emoji} {signal}</p>
                    <p style='color: #e0e0e0; margin: 0;'><strong>{result["confidence"]}</strong> {result["conf_emoji"]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability Grid [1, 3]
            st.markdown("#### Signal Probabilities")
            prob_cols = st.columns([1,1,1,1.5], gap="small")
            
            with prob_cols[0]:
                prob_container_buy = st.container(border=True)
                with prob_container_buy:
                    st.metric("**Buy**", f"{result['buy_pct']:.1f}%")
            
            with prob_cols[1]:
                prob_container_hold = st.container(border=True)
                with prob_container_hold:
                    st.metric("**Hold**", f"{result['hold_pct']:.1f}%")
            
            with prob_cols[2]:
                prob_container_sell = st.container(border=True)
                with prob_container_sell:
                    st.metric("**Sell**", f"{result['sell_pct']:.1f}%")
            
            # Market Regime
            regime_map = {0: 'BULLISH', 1: 'BEARISH', 2: 'HIGH-VOLATILITY'}
            regime_name = regime_map.get(regime_code, 'UNKNOWN')
            
            with prob_cols[3]:
                regime_container = st.container(border=True)
                with regime_container:
                    delta_buy = result.get('delta_buy', 'N/A')
                    delta_str = f"{delta_buy:.1f}%" if delta_buy != 'N/A' else 'N/A'
                    tip = f"Base signal: {result['signal_no_reg']} | Regime impact: {delta_str}"
                    
                    st.metric("**Market Regime**", regime_name, help=tip)
            
                
            # Historical & Predicted Price Chart
            
            days_for_chart = HORIZON_MAP.get(st.session_state.selected_horizon, 180)
            
            pred_trend_data = predict_past_buy_probability(ticker, prediction_date, days_ahead=days_for_chart)
            
            if pred_trend_data:
                hist_container = st.container(border=True)
                
                with hist_container:
                    st.markdown("#### Stock Trend Analysis")
                    # Historical Price Chart
                    price_chart = plot_stock_chart(ticker, prediction_date, days=days_for_chart)
                    
                    if price_chart:
                        price_chart.add_vline(x=prediction_date, line_dash="solid", line_color="red", opacity=0.7)
                        st.plotly_chart(price_chart, width='stretch')
                    else:
                        st.warning(f"Could not retrieve price data for {ticker}.")
                    chart_df = pd.DataFrame({
                        'Date': pred_trend_data['past_dates'],
                        'BUY Probability': pred_trend_data['past_predictions']
                    })
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['BUY Probability'],
                        mode='lines',
                        name='Model Prediction',
                        line=dict(color='#0084ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 132, 255, 0.1)'
                    ))

                    
                    # Model Buy Probability Trend
                    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
                    fig.add_vline(x=prediction_date, line_dash="solid", line_color="red", opacity=0.7)
                    
                    fig.update_layout(
                        title=f'{ticker} - Model Buy Probability Trend (Last {days_for_chart} Days)',
                        xaxis_title='Date',
                        yaxis_title='BUY Probability (%)',
                        hovermode='x unified',
                        template='plotly_dark',
                        height=400,
                        plot_bgcolor='rgba(15, 20, 25, 0.5)',
                        paper_bgcolor='rgba(15, 20, 25, 0)',
                        font=dict(color='#e0e0e0', size=11)
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    
            
            else:
                st.warning("Could not generate trend analysis.")
