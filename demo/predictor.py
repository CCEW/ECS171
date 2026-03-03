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
    'Ret_Lag1', 'Ret_Lag2', 'RSI', 'MACD_Norm',
    'ATR_Pct', 'HL_range', 'Price_vs_SMA50',
    'Momentum_Deviation', 'BB_Pct'
]
# S2 = S1_FEATURES + [Encoded_Ticker, regime_pred]  (11 total)

# ── S&P 500 ticker → label-encoded ID (alphabetical, matches training) ────────
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


class StockPredictor:
    """
    Two-stage regime-aware stock signal predictor.

    Usage:
        predictor = StockPredictor('models/')
        info = predictor.get_info('NVDA', '2026-02-26') i.e (Ticker, Date)
        regime_code, result = predictor.run_prediction(info)
    """

    SIGNAL_MAP  = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    REGIME_MAP  = {0: 'BULLISH', 1: 'BEARISH', 2: 'HIGH-VOLATILITY'}
    WARMUP_DAYS = 120

    def __init__(self, model_dir: str = 'models/'):
        """Load all four pkl files from model_dir."""
        def _load(fname):
            with open(f'{model_dir}/{fname}', 'rb') as f:
                return pickle.load(f)

        self.lgb_s1 = _load('lgb_s1.pkl')
        self.lgb_s2 = _load('lgb_s2.pkl')
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
        Compute all 9 technical features from OHLC data.
        Requires 60+ rows of history for accurate rolling windows.
        """
        df = df_raw.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']

        df['Ret_Lag1'] = close.pct_change(1)
        df['Ret_Lag2'] = close.pct_change(2).shift(1)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal    = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_Norm'] = (macd_line - signal) / close

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['ATR_Pct'] = tr.rolling(14).mean() / close

        df['HL_range'] = (high - low) / close

        df['Price_vs_SMA50'] = close / close.rolling(50).mean()

        sma20 = close.rolling(20).mean()
        df['Momentum_Deviation'] = (close - sma20) / sma20

        bb_std = close.rolling(20).std()
        bb_up  = sma20 + 2 * bb_std
        bb_low = sma20 - 2 * bb_std
        df['BB_Pct'] = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)

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
            encoded_ticker  int   (250 if unknown ticker)
            feat_vals np.ndarray shape (9,)  or None on error
            error str | None
        """
        ticker = ticker.strip().upper()
        target_date = pd.to_datetime(target_date_str)
        start_date  = target_date - timedelta(days=self.WARMUP_DAYS)
        end_date = target_date + timedelta(days=1)

        encoded_ticker = TICKER_ENCODING.get(ticker, 250)

        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return {
                'ticker': ticker, 'encoded_ticker': encoded_ticker,
                'feat_vals': None,
                'error': f'No data returned by yfinance for {ticker}'
            }

        raw = self._flatten_yf_columns(raw)

        # 3. Compute features
        feat = self._compute_features(raw)
        target_row = feat[feat.index.date == target_date.date()]

        if target_row.empty:
            return {
                'ticker': ticker, 'encoded_ticker': encoded_ticker,
                'feat_vals': None,
                'error': f'No trading data on {target_date_str} — market may be closed'
            }

        feat_vals = target_row[S1_FEATURES].values.flatten()

        if np.any(np.isnan(feat_vals)):
            return {
                'ticker': ticker, 'encoded_ticker': encoded_ticker,
                'feat_vals': None,
                'error': 'NaN in computed features — insufficient history for rolling windows'
            }

        return {
            'ticker': ticker,
            'encoded_ticker': encoded_ticker,
            'feat_vals': feat_vals,
            'error': None,
        }

    def run_prediction(self, info: dict) -> tuple:
        """
        Runs Stage 1 (regime) and Stage 2 (Buy/Hold/Sell) on the output of get_info.

        Parameters:
            info dict returned by get_info

        Returns:
            regime_code  int   0=Bullish | 1=Bearish | 2=High-Volatility
            result dict  {
                ticker, signal, sell_pct, hold_pct, buy_pct,
                confidence, conf_emoji, signal_no_reg, delta_buy, error
            }
        """
        if info['error']: return 0, {'ticker': info['ticker'], 'error': info['error']}

        feat_vals = info['feat_vals']
        encoded_ticker = info['encoded_ticker']

        regime_code = 0
        try:
            X_s1 = self.scaler_s1.transform(feat_vals.reshape(1, -1))
            regime_code = int(self.lgb_s1.predict(X_s1)[0])
        except Exception:
            pass

        X_s2_regime    = np.append(feat_vals, [encoded_ticker, regime_code]).reshape(1, -1)
        X_s2_regime_sc = self.scaler_s2.transform(X_s2_regime)
        proba_regime   = self.lgb_s2.predict_proba(X_s2_regime_sc)[0]

        X_s2_base    = np.append(feat_vals, [encoded_ticker, 0]).reshape(1, -1)
        X_s2_base_sc = self.scaler_s2.transform(X_s2_base)
        proba_base   = self.lgb_s2.predict_proba(X_s2_base_sc)[0]

        signal = self.SIGNAL_MAP[int(np.argmax(proba_regime))]
        signal_base = self.SIGNAL_MAP[int(np.argmax(proba_base))]
        
        sorted_proba = np.sort(proba_regime)[::-1]
        margin = sorted_proba[0] - sorted_proba[1]

        if margin >= 0.08: confidence, conf_emoji = 'High',   '🔵'
        elif margin >= 0.04: confidence, conf_emoji = 'Medium', '🟡'
        else: confidence, conf_emoji = 'Low',    '🔴'

        result = {
            'ticker': info['ticker'],
            'signal': signal,
            'sell_pct': round(proba_regime[0] * 100, 1),
            'hold_pct': round(proba_regime[1] * 100, 1),
            'buy_pct': round(proba_regime[2] * 100, 1),
            'confidence': confidence,
            'conf_emoji': conf_emoji,
            'signal_no_reg': signal_base,
            'delta_buy': round((proba_regime[2] - proba_base[2]) * 100, 2),
            'error': None,
        }

        return regime_code, result
