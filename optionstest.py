import streamlit as st
import datetime as dt
import scipy.stats as si
import scipy.interpolate
from scipy.stats import linregress
import pandas as pd
import requests
import numpy as np
import ccxt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d, CubicSpline
import logging
import time
from plotly.subplots import make_subplots
import math
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
BASE_URL = "https://thalex.com/api/v2/public"
INSTRUMENTS_ENDPOINT = "instruments"
URL_INSTRUMENTS = f"{BASE_URL}/{INSTRUMENTS_ENDPOINT}"
MARK_PRICE_ENDPOINT = "mark_price_historical_data"
URL_MARK_PRICE = f"{BASE_URL}/{MARK_PRICE_ENDPOINT}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
COLUMNS = ["ts", "mark_price_open", "mark_price_high", "mark_price_low","mark_price_close", "iv_open", "iv_high", "iv_low", "iv_close","mark_volume" ]
REQUEST_TIMEOUT = 15
TRANSACTION_COST_BPS = 2 

# --- Utility Functions ---

## Login Functions
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            users = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            pwds = [line.strip() for line in f_pass if line.strip()]
        if len(users) != len(pwds):
            st.error("Number of usernames and passwords mismatch.")
            return {}
        return dict(zip(users, pwds))
    except FileNotFoundError:
        st.error("Credential files (usernames.txt, passwords.txt) not found.")
        return {}
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Please Log In")
        creds = load_credentials()
        if not creds: st.stop()
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.rerun()
            else: st.error("Invalid username or password")
        st.stop()

def safe_get_in(keys, data_dict, default=None):
    current = data_dict
    for key in keys:
        if isinstance(current, dict) and key in current: current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current): current = current[key]
        else: return default
    return current

## Fetch and Filter Functions
@st.cache_data(ttl=600)
def fetch_instruments():
    try:
        resp = requests.get(URL_INSTRUMENTS, timeout=REQUEST_TIMEOUT); resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e: st.error(f"Error fetching instruments: {e}"); return []

@st.cache_data(ttl=30)
def fetch_ticker(instr_name):
    try:
        r = requests.get(URL_TICKER, params={"instrument_name": instr_name}, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        return r.json().get("result", {})
    except Exception as e: logging.warning(f"Error fetching ticker {instr_name}: {e}"); return None

def params_historical(instrument_name, days=7): # days parameter added here
    now = dt.datetime.now(dt.timezone.utc); start_dt = now - dt.timedelta(days=days)
    return {"from": int(start_dt.timestamp()), "to": int(now.timestamp()), "resolution": "5m", "instrument_name": instrument_name}

@st.cache_data(ttl=60)
def fetch_data(instruments_tuple, days_history=7): # Added days_history parameter
    instr = list(instruments_tuple)
    if not instr: return pd.DataFrame()
    dfs = []
    logging.info(f"Fetching data for {len(instr)} instruments with {days_history} days history.")
    for name in instr:
        try:
            # Use days_history in params_historical
            resp = requests.get(URL_MARK_PRICE, params=params_historical(name, days=days_history), timeout=REQUEST_TIMEOUT); resp.raise_for_status()
            marks = safe_get_in(["result", "mark"], resp.json(), default=[])
            if marks:
                df_temp = pd.DataFrame(marks, columns=COLUMNS); df_temp["instrument_name"] = name
                if not df_temp.empty and "ts" in df_temp.columns and df_temp["ts"].notna().any(): dfs.append(df_temp)
        except Exception as e: logging.error(f"Error fetching data for {name}: {e}")
        time.sleep(0.05) 
    if not dfs: 
        logging.warning(f"fetch_data: No dataframes collected for instruments after fetch loop (days_history={days_history}).")
        return pd.DataFrame()
    try:
        dfc = pd.concat(dfs).reset_index(drop=True)
        dfc['date_time'] = pd.to_datetime(dfc['ts'], unit='s', errors='coerce').dt.tz_localize('UTC')
        dfc = dfc.dropna(subset=['date_time'])
        dfc['k'] = dfc['instrument_name'].apply(lambda s: int(s.split('-')[2]) if isinstance(s, str) and len(s.split('-'))>2 and s.split('-')[2].isdigit() else np.nan)
        dfc['option_type'] = dfc['instrument_name'].apply(lambda s: s.split('-')[-1] if isinstance(s, str) and len(s.split('-'))>2 else None)
        dfc['expiry_datetime_col'] = dfc['instrument_name'].apply(lambda s: dt.datetime.strptime(s.split('-')[1], "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8) if isinstance(s, str) and len(s.split('-'))>1 else pd.NaT)
        
        # Critical: Ensure iv_close is numeric before dropping NaNs based on it
        dfc['iv_close'] = pd.to_numeric(dfc['iv_close'], errors='coerce')
        
        initial_rows = len(dfc)
        dfc = dfc.dropna(subset=['k', 'option_type', 'mark_price_close', 'iv_close', 'expiry_datetime_col']) 
        rows_dropped = initial_rows - len(dfc)
        logging.info(f"fetch_data: Dropped {rows_dropped} rows due to NaNs in critical columns (incl. iv_close). {len(dfc)} rows remaining.")
        
        return dfc.sort_values("date_time") if not dfc.empty else pd.DataFrame()
    except Exception as e: st.error(f"Error processing fetched data: {e}"); return pd.DataFrame()

exchange1 = None
try: exchange1 = ccxt.bitget({'enableRateLimit': True})
except Exception as e: st.error(f"Failed to connect to Bitget: {e}")

def fetch_funding_rates(exchange_instance, symbol='BTC/USDT', start_time=None, end_time=None):
    if exchange_instance is None: return pd.DataFrame()
    try:
        markets = exchange_instance.load_markets()
        if symbol not in markets: return pd.DataFrame()
        since = int(start_time.timestamp() * 1000) if start_time else None
        hist = exchange_instance.fetch_funding_rate_history(symbol=symbol, since=since)
        if not hist: return pd.DataFrame()
        data = [{'date_time': pd.to_datetime(e['timestamp'], unit='ms', utc=True), 
                 'raw_funding_rate': e['fundingRate'], 
                 'funding_rate': e['fundingRate'] * 365 * 3} 
                for e in hist if not (end_time and pd.to_datetime(e['timestamp'], unit='ms', utc=True) > end_time)]
        return pd.DataFrame(data)
    except Exception as e: logging.error(f"Error fetching funding rates: {e}"); return pd.DataFrame()

def fetch_kraken_data(coin="BTC", days=7, timeframe="5m"): # Keep kraken data fetch separate, as it's for spot
    try:
        k = ccxt.kraken(); now = dt.datetime.now(dt.timezone.utc); start = now - dt.timedelta(days=days)
        ohlcv = k.fetch_ohlcv(f"{coin}/USD", timeframe=timeframe, since=int(start.timestamp()*1000))
        if not ohlcv: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date_time"] = pd.to_datetime(df["timestamp"], unit="ms", errors='coerce').dt.tz_localize("UTC")
        return df.dropna(subset=['date_time']).sort_values("date_time").reset_index(drop=True)
    except Exception as e: st.error(f"Error fetching Kraken {timeframe} data: {e}"); return pd.DataFrame()

def get_valid_expiration_options(current_date_utc):
    instruments = fetch_instruments()
    if not instruments: return []
    exp_dates = set()
    for instr in instruments:
        try:
            parts = instr.get("instrument_name", "").split("-")
            if len(parts) >= 3 and parts[-1] in ['C', 'P']:
                expiry_date = dt.datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
                if expiry_date > current_date_utc: exp_dates.add(expiry_date)
        except Exception: continue
    return sorted(list(exp_dates))

def get_option_instruments(instruments, option_type, expiry_str, coin):
    return sorted([i["instrument_name"] for i in instruments if i.get("instrument_name","").startswith(f"{coin}-{expiry_str}") and i.get("instrument_name","").endswith(f"-{option_type}")])

# --- Greek Calculations (Unchanged) ---
def compute_delta(row, S, snapshot_time_utc, r=0.0):
    try:
        k, sigma, option_type, expiry_date = row['k'], row['iv_close'], row['option_type'], row['expiry_datetime_col']
        if pd.isna(S) or S<=0 or pd.isna(k) or k<=0 or pd.isna(sigma) or sigma<=0 or pd.isna(option_type) or pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365*24*3600)
        if T < 1e-9: return 1.0 if option_type == 'C' and S > k else (-1.0 if option_type == 'P' and S < k else 0.0)
        sigma_sqrt_T = sigma*np.sqrt(T)
        if abs(sigma_sqrt_T) < 1e-12: return 1.0 if option_type == 'C' and S > k else (-1.0 if option_type == 'P' and S < k else 0.0) 
        d1 = (np.log(S/k) + (r + 0.5*sigma**2)*T) / sigma_sqrt_T
        if not np.isfinite(d1): return np.nan 
        return norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1.0
    except Exception: return np.nan

def compute_gamma(row, S, snapshot_time_utc, r=0.0):
    try:
        k, sigma, expiry_date = row['k'], row['iv_close'], row['expiry_datetime_col']
        if pd.isna(S) or S<=0 or pd.isna(k) or k<=0 or pd.isna(sigma) or sigma<=0 or pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365*24*3600)
        sigma_sqrt_T = sigma*np.sqrt(T)
        if T < 1e-9 or abs(sigma_sqrt_T) < 1e-12 or abs(S * sigma_sqrt_T) < 1e-12: return 0.0
        d1 = (np.log(S/k) + (r + 0.5*sigma**2)*T) / sigma_sqrt_T
        if not np.isfinite(d1): return 0.0
        return norm.pdf(d1) / (S*sigma_sqrt_T)
    except Exception: return np.nan

def compute_vega(row, S, snapshot_time_utc, r=0.0):
    try:
        k, sigma, expiry_date = row['k'], row['iv_close'], row['expiry_datetime_col']
        if pd.isna(S) or S<=0 or pd.isna(k) or k<=0 or pd.isna(sigma) or sigma<=0 or pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365*24*3600)
        sigma_sqrt_T = sigma*np.sqrt(T)
        if T < 1e-9 or abs(sigma_sqrt_T) < 1e-12 : return 0.0
        d1 = (np.log(S/k) + (r + 0.5*sigma**2)*T) / sigma_sqrt_T
        if not np.isfinite(d1): return 0.0
        return S * norm.pdf(d1) * np.sqrt(T) * 0.01 
    except Exception: return np.nan

def compute_charm(row, S, snapshot_time_utc, r=0.0): 
    try:
        k, sigma, option_type, expiry_date = row['k'], row['iv_close'], row['option_type'], row['expiry_datetime_col']
        if pd.isna(S) or S<=0 or pd.isna(k) or k<=0 or pd.isna(sigma) or sigma<=0 or pd.isna(option_type) or pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365*24*3600)
        sigma_sqrt_T = sigma*np.sqrt(T)
        if T < 1e-9 or abs(sigma_sqrt_T) < 1e-12 or abs(2*T) < 1e-12 : return 0.0
        d1 = (np.log(S/k) + (r + 0.5*sigma**2)*T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        if not (np.isfinite(d1) and np.isfinite(d2)): return 0.0
        charm_annual = norm.pdf(d1) * d2 / (2*T) 
        return charm_annual / 365.0 
    except Exception: return np.nan

def compute_vanna(row, S, snapshot_time_utc, r=0.0): 
    try:
        k, sigma, expiry_date = row['k'], row['iv_close'], row['expiry_datetime_col']
        if pd.isna(S) or S<=0 or pd.isna(k) or k<=0 or pd.isna(sigma) or sigma<=0 or pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365*24*3600)
        sigma_sqrt_T = sigma*np.sqrt(T)
        if T < 1e-9 or abs(sigma_sqrt_T) < 1e-12 or abs(sigma) < 1e-12: return 0.0
        d1 = (np.log(S/k) + (r + 0.5*sigma**2)*T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        if not (np.isfinite(d1) and np.isfinite(d2)): return 0.0
        vanna = -norm.pdf(d1) * d2 / sigma 
        return vanna * 0.01 
    except Exception: return np.nan

def compute_gex(row, S, oi): 
    try:
        gamma_val, oi_val = row.get('gamma'), float(oi) if pd.notna(oi) else np.nan
        if pd.isna(gamma_val) or pd.isna(oi_val) or pd.isna(S) or S <= 0 or oi_val < 0: return np.nan
        return gamma_val * oi_val * (S ** 2) * 0.01 
    except Exception: return np.nan
# --- Supporting View Plotting Functions (Unchanged) ---
def build_ticker_list(dft_latest, ticker_data):
    if dft_latest.empty: return []
    required_cols = ['instrument_name', 'k', 'option_type', 'delta', 'gamma', 'open_interest']
    if not all(c in dft_latest.columns for c in required_cols): return []
    tl = []
    for _, row in dft_latest.iterrows():
        instr = row['instrument_name']; td = ticker_data.get(instr)
        if not td or td.get('iv') is None: continue
        if any(pd.isna(row[c]) for c in ['delta', 'gamma', 'k', 'open_interest']): continue
        try: tl.append({"instrument": instr, "strike": int(row['k']), "option_type": row['option_type'], "open_interest": float(row['open_interest']), "delta": float(row['delta']), "gamma": float(row['gamma']), "iv": float(td['iv'])})
        except (TypeError, ValueError): continue
    return sorted(tl, key=lambda x: x['strike'])

def plot_delta_balance(ticker_list, spot_price):
    st.subheader("Put vs Call Delta Balance (OI Weighted)")
    if not ticker_list or pd.isna(spot_price): st.info("Data insufficient for Delta Balance plot."); return
    try:
        calls_delta = sum(i["delta"] * i["open_interest"] for i in ticker_list if i["option_type"] == "C" and pd.notna(i["delta"]) and pd.notna(i["open_interest"]))
        puts_delta = sum(i["delta"] * i["open_interest"] for i in ticker_list if i["option_type"] == "P" and pd.notna(i["delta"]) and pd.notna(i["open_interest"]))
        data = pd.DataFrame({'Option Type': ['Calls', 'Puts'], 'Total Weighted Delta': [calls_delta, abs(puts_delta)]})
        fig = px.bar(data, x='Option Type', y='Total Weighted Delta', color='Option Type', color_discrete_map={"Calls": "mediumseagreen", "Puts": "lightcoral"}, title='Put vs Call Delta Balance (OI Weighted)', labels={'Total Weighted Delta': 'Absolute Total Delta * OI'})
        net_delta = calls_delta + puts_delta 
        fig.add_annotation(text=f"Net Delta Exposure (Market): {net_delta:,.2f}", align='center', showarrow=False, xref='paper', yref='paper', x=0.5, y=1.05, font=dict(size=14, color="blue" if net_delta > 0 else "red" if net_delta < 0 else "grey"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error plotting Delta Balance: {e}")

def plot_open_interest_delta(ticker_list, spot_price):
    st.subheader("Open Interest & Delta Bubble Chart")
    if not ticker_list or pd.isna(spot_price): st.info("Data insufficient for OI & Delta Bubble chart."); return
    try:
        df = pd.DataFrame(ticker_list).dropna(subset=['strike', 'open_interest', 'delta', 'iv'])
        if df.empty: st.info("No valid data for OI & Delta Bubble chart."); return
        df['moneyness'] = df['strike'] / spot_price
        fig = px.scatter(df, x="strike", y="delta", size="open_interest", color="moneyness", color_continuous_scale=px.colors.diverging.RdYlBu_r, range_color=[0.8, 1.2], hover_data=["instrument", "open_interest", "iv"], size_max=50, title=f"Open Interest & Delta by Strike (Size=OI, Color=Moneyness vs Spot={spot_price:.0f})")
        fig.add_vline(x=spot_price, line_dash="dot", line_color="black", annotation_text="Spot")
        fig.add_hline(y=0.5, line_dash="dot", line_color="grey"); fig.add_hline(y=-0.5, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error plotting OI & Delta Bubble chart: {e}")

def filter_df_by_strike_range(df, spot_price, t_years, multiplier):
    if df.empty or 'k' not in df.columns or 'iv_close' not in df.columns: return df
    if multiplier == float('inf') or pd.isna(spot_price) or pd.isna(t_years): return df
    avg_iv = df['iv_close'].mean(); avg_iv = 0.5 if pd.isna(avg_iv) or avg_iv <=0 else avg_iv
    try:
        exp_term = avg_iv * np.sqrt(max(1e-4, t_years)) * multiplier
        lo, hi = spot_price * np.exp(-exp_term), spot_price * np.exp(exp_term)
        return df[(df['k'] >= lo) & (df['k'] <= hi)].copy()
    except Exception: return df
# --- ITM GEX Analysis Functions (Unchanged) ---
def compute_and_plot_itm_gex_ratio(dft, df_krak_5m, spot_price_latest, selected_expiry_obj):
    expiry_label = selected_expiry_obj.strftime('%d%b%y') if isinstance(selected_expiry_obj, dt.datetime) else "N/A"
    coin = dft['instrument_name'].iloc[0].split('-')[0] if not dft.empty and 'instrument_name' in dft.columns else "N/A"
    st.subheader(f"ITM Put/Call GEX Ratio (Expiry: {expiry_label})")
    if dft.empty or df_krak_5m.empty: st.info("GEX Ratio: Data insufficient."); return pd.DataFrame()
    df_merged_gex = dft.copy() 
    if 'spot_hist' not in df_merged_gex.columns and 'spot_price' not in df_merged_gex.columns : 
        df_merged_gex = pd.merge_asof(dft.sort_values('date_time'), df_krak_5m[['date_time','close']].rename(columns={'close':'spot_price'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('10min')).dropna(subset=['spot_price'])
    elif 'spot_hist' in df_merged_gex.columns and 'spot_price' not in df_merged_gex.columns:
         df_merged_gex = df_merged_gex.rename(columns={'spot_hist': 'spot_price'})
    if df_merged_gex.empty or 'spot_price' not in df_merged_gex.columns: st.info("GEX Ratio: No data after spot merge/check."); return pd.DataFrame()
    risk_free_rate = st.session_state.get('risk_free_rate_input', 0.0)
    if 'gamma' not in df_merged_gex.columns:
        with st.spinner(f"Recalculating Gamma for GEX (Expiry: {expiry_label})..."):
             df_merged_gex['gamma'] = df_merged_gex.apply(lambda row: compute_gamma(row, row['spot_price'], row['date_time'], risk_free_rate), axis=1)
    if 'open_interest' not in df_merged_gex.columns : 
        st.warning("Open interest column missing for GEX calculation.")
        df_merged_gex['open_interest'] = 1 
    df_merged_gex['gex'] = df_merged_gex.apply(lambda row: compute_gex(row, row['spot_price'], row['open_interest']), axis=1)
    df_merged_gex = df_merged_gex.dropna(subset=['gex','k','option_type','spot_price']) 
    if df_merged_gex.empty: st.info("GEX Ratio: No GEX data after calculation."); return pd.DataFrame()
    df_merged_gex['is_itm_call'] = (df_merged_gex['option_type'] == 'C') & (df_merged_gex['k'] < df_merged_gex['spot_price'])
    df_merged_gex['is_itm_put'] = (df_merged_gex['option_type'] == 'P') & (df_merged_gex['k'] > df_merged_gex['spot_price'])
    df_agg = df_merged_gex.groupby('date_time').apply(lambda g: pd.Series({'total_itm_call_gex': g.loc[g['is_itm_call'], 'gex'].sum(skipna=True),'total_itm_put_gex': g.loc[g['is_itm_put'], 'gex'].sum(skipna=True)}), include_groups=False).reset_index()
    epsilon = 1e-9
    df_agg['itm_gex_ratio'] = df_agg['total_itm_put_gex'] / (df_agg['total_itm_call_gex'] + epsilon)
    df_agg['itm_gex_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_plot = df_agg.dropna(subset=['itm_gex_ratio'])
    if df_plot.empty or len(df_plot) < 2: st.info("GEX Ratio: Not enough data to plot ratio."); return pd.DataFrame()
    df_plot = pd.merge_asof(df_plot.sort_values('date_time'), df_krak_5m[['date_time', 'close']].sort_values('date_time'), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min')).dropna(subset=['close'])
    if df_plot.empty: st.info("GEX Ratio: No data after final merge for plotting."); return pd.DataFrame()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['itm_gex_ratio'], name='ITM Put/Call GEX Ratio', mode='lines', line=dict(color='mediumseagreen')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['close'], name=f'{coin} Spot Price', mode='lines', line=dict(color='cornflowerblue')), secondary_y=True)
    title_text = f"{coin} Intraday - ${spot_price_latest:,.2f}, ITM PUT/CALL GEX Ratio (Exp: {expiry_label})"
    fig.update_layout(title=title_text, height=500, xaxis_title="Date (UTC)", yaxis_title="Ratio", yaxis2_title="Price", legend_title_text="Metric", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    if not df_plot.empty and 'itm_gex_ratio' in df_plot.columns and df_plot['itm_gex_ratio'].notna().any() : st.metric("Latest ITM P/C GEX Ratio", f"{df_plot['itm_gex_ratio'].iloc[-1]:.2f}")
    return df_plot
# --- Net Delta OTM Pair Sim (Unchanged, but benefits from better data via configurable lookback) ---
def plot_net_delta_otm_pair(dft: pd.DataFrame, 
                            df_spot_hist: pd.DataFrame, 
                            exchange_instance, 
                            selected_call_instr: str, 
                            selected_put_instr: str,
                            all_available_calls_for_expiry: list, 
                            all_available_puts_for_expiry: list 
                           ):
    function_name = "plot_net_delta_otm_pair"
    logging.info(f"{function_name}: Starting with initial selections: Call={selected_call_instr}, Put={selected_put_instr}")

    if dft.empty or df_spot_hist.empty:
        st.info("Historical option or spot data is insufficient for pair simulation.")
        logging.info(f"{function_name}: dft or df_spot_hist empty at start.")
        return

    def prep_df(df_opt_orig, strike_val, option_t_char, df_name_log, instr_name_for_log):
        logging.info(f"{function_name}: prep_df starting for {df_name_log} ({instr_name_for_log}). Input original shape: {df_opt_orig.shape if not df_opt_orig.empty else 'N/A'}")
        if df_opt_orig.empty:
             logging.warning(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}): Input df_opt_orig is empty.")
             return pd.DataFrame()
        if 'iv_close' not in df_opt_orig.columns:
            logging.error(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}): 'iv_close' missing from df_opt_orig.")
            return pd.DataFrame()
        df_opt_orig['iv_close'] = pd.to_numeric(df_opt_orig['iv_close'], errors='coerce') 
        
        if not all(col in df_opt_orig.columns for col in ['date_time', 'iv_close', 'expiry_datetime_col']): 
            logging.error(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}): Input df_opt_orig missing required columns. Has: {df_opt_orig.columns.tolist()}")
            return pd.DataFrame()
        
        df_m = pd.merge_asof(df_opt_orig.sort_values('date_time'), 
                             df_spot_hist[['date_time','close']].rename(columns={'close':'spot_hist'}), 
                             on='date_time', direction='nearest', tolerance=pd.Timedelta('15min'))
        
        base_asset_for_funding = "BTC" 
        if not dft.empty and 'instrument_name' in dft.columns and len(dft['instrument_name'].iloc[0].split('-')) > 0:
            base_asset_for_funding = dft['instrument_name'].iloc[0].split('-')[0]
        
        df_funding = fetch_funding_rates(exchange_instance, symbol=f"{base_asset_for_funding}/USDT", 
                                        start_time=df_opt_orig['date_time'].min() if not df_opt_orig.empty else None, 
                                        end_time=df_opt_orig['date_time'].max() if not df_opt_orig.empty else None)

        if not df_funding.empty and 'date_time' in df_funding.columns:
            df_m = pd.merge_asof(df_m.sort_values('date_time'), 
                                 df_funding[['date_time','funding_rate']], 
                                 on='date_time', direction='nearest', tolerance=pd.Timedelta('8hour'))
        else: df_m['funding_rate'] = 0.0
        df_m['funding_rate'] = df_m['funding_rate'].fillna(0.0)
        
        df_m['k'] = strike_val 
        df_m['option_type'] = option_t_char
        
        required_subset_for_dropna = ['spot_hist', 'iv_close', 'k', 'option_type', 'funding_rate', 'date_time', 'expiry_datetime_col']
        missing_cols_for_dropna = [col for col in required_subset_for_dropna if col not in df_m.columns]
        if missing_cols_for_dropna:
            logging.error(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}): df_m missing columns before dropna: {missing_cols_for_dropna}")
            return pd.DataFrame()

        df_m_cleaned = df_m.dropna(subset=required_subset_for_dropna + ['iv_close']) 
        logging.info(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}) after dropna. Shape: {df_m_cleaned.shape}. Original input IV non-NaN count: {df_opt_orig['iv_close'].notna().sum()}")
        if df_m_cleaned.empty:
            logging.warning(f"{function_name}: prep_df for {df_name_log} ({instr_name_for_log}) is empty AFTER dropna. Check 'iv_close' and other required fields for NaNs.")
        return df_m_cleaned

    current_call_instr = selected_call_instr
    current_put_instr = selected_put_instr
    
    if not current_call_instr:
        st.warning("No initial call instrument selected for pair sim."); logging.warning(f"{function_name}: No initial call."); return
    try: call_strike = int(current_call_instr.split('-')[-2])
    except: st.error(f"Invalid call instrument name: {current_call_instr}"); logging.error(f"{function_name}: Invalid call name {current_call_instr}"); return
    
    df_call_orig = dft[dft['instrument_name'] == current_call_instr].copy()
    df_call = prep_df(df_call_orig, call_strike, 'C', "INITIAL_CALL", current_call_instr)

    if df_call.empty:
        st.warning(f"Initial Call ({current_call_instr}) data empty after prep. Attempting fallback...")
        logging.warning(f"{function_name}: Initial call {current_call_instr} resulted in empty df. Fallback initiated.")
        for alt_call_instr in all_available_calls_for_expiry:
            if alt_call_instr == current_call_instr: continue 
            try: alt_call_strike = int(alt_call_instr.split('-')[-2])
            except: continue 
            
            alt_df_call_orig = dft[dft['instrument_name'] == alt_call_instr].copy()
            df_call_alt = prep_df(alt_df_call_orig, alt_call_strike, 'C', "ALT_CALL", alt_call_instr)
            if not df_call_alt.empty:
                st.warning(f"Using fallback Call: {alt_call_instr}")
                logging.info(f"{function_name}: Fallback Call successful: {alt_call_instr}")
                current_call_instr = alt_call_instr
                call_strike = alt_call_strike
                df_call = df_call_alt
                break 
        if df_call.empty: 
            st.error(f"Could not find any valid Call option with data for pair simulation after fallback attempts. This usually means no historical IV data was found for any calls of this expiry within the selected lookback period."); 
            logging.error(f"{function_name}: Fallback for call failed."); return

    if not current_put_instr:
        st.warning("No initial put instrument selected for pair sim."); logging.warning(f"{function_name}: No initial put."); return
    try: put_strike = int(current_put_instr.split('-')[-2])
    except: st.error(f"Invalid put instrument name: {current_put_instr}"); logging.error(f"{function_name}: Invalid put name {current_put_instr}"); return

    df_put_orig = dft[dft['instrument_name'] == current_put_instr].copy()
    df_put = prep_df(df_put_orig, put_strike, 'P', "INITIAL_PUT", current_put_instr)

    if df_put.empty:
        st.warning(f"Initial Put ({current_put_instr}) data empty after prep. Attempting fallback...")
        logging.warning(f"{function_name}: Initial put {current_put_instr} resulted in empty df. Fallback initiated.")
        for alt_put_instr in all_available_puts_for_expiry:
            if alt_put_instr == current_put_instr: continue
            try: alt_put_strike = int(alt_put_instr.split('-')[-2])
            except: continue
            
            alt_df_put_orig = dft[dft['instrument_name'] == alt_put_instr].copy()
            df_put_alt = prep_df(alt_df_put_orig, alt_put_strike, 'P', "ALT_PUT", alt_put_instr)
            if not df_put_alt.empty:
                st.warning(f"Using fallback Put: {alt_put_instr}")
                logging.info(f"{function_name}: Fallback Put successful: {alt_put_instr}")
                current_put_instr = alt_put_instr
                put_strike = alt_put_strike
                df_put = df_put_alt
                break
        if df_put.empty: 
            st.error(f"Could not find any valid Put option with data for pair simulation after fallback attempts. This usually means no historical IV data was found for any puts of this expiry within the selected lookback period."); 
            logging.error(f"{function_name}: Fallback for put failed."); return
            
    st.subheader(f"Net Delta Sim (as Perpetuals): Short {current_call_instr.split('-')[-2]}C / {current_put_instr.split('-')[-2]}P")

    try:
        with st.spinner(f"Calculating perpetual-like deltas for {current_call_instr}..."):
            df_call['delta_calc'] = df_call.apply(lambda r: compute_delta1(r, r['spot_hist'], r['date_time'], r['funding_rate'], is_perpetual=True), axis=1)
        with st.spinner(f"Calculating perpetual-like deltas for {current_put_instr}..."):
            df_put['delta_calc'] = df_put.apply(lambda r: compute_delta1(r, r['spot_hist'], r['date_time'], r['funding_rate'], is_perpetual=True), axis=1)
    except KeyError as e: st.error(f"KeyError during delta calculation for pair sim. Missing column: {e}."); return
    except Exception as e: st.error(f"Unexpected error during delta calculation for pair sim: {e}"); return

    df_call = df_call.dropna(subset=['delta_calc'])
    df_put = df_put.dropna(subset=['delta_calc'])

    if df_call.empty: st.warning(f"No valid deltas for (fallback) Call: {current_call_instr}"); return
    if df_put.empty: st.warning(f"No valid deltas for (fallback) Put: {current_put_instr}"); return

    df_combined = pd.merge(
        df_call[['date_time', 'delta_calc', 'spot_hist']].rename(columns={'delta_calc':'delta_call'}), 
        df_put[['date_time', 'delta_calc']], 
        on='date_time', how='inner'
    ).dropna().sort_values('date_time').reset_index(drop=True)

    if df_combined.empty: st.info("No combined data after merging call and put deltas for pair sim."); return
    if 'spot_hist' not in df_combined.columns: st.error("'spot_hist' column missing from df_combined."); return
    if df_combined['spot_hist'].isna().all(): st.error("All 'spot_hist' values NaN in df_combined."); return

    df_combined['option_delta_total'] = -(df_combined['delta_call'] + df_combined['delta_put']) 
    hedge_pos_delta_val = 0.0; history = []; HEDGE_THRESHOLD = 0.05; TARGET_NET_DELTA = 0.0
    for _, row in df_combined.iterrows():
        current_portfolio_delta = row['option_delta_total'] + hedge_pos_delta_val
        hedge_amount_needed_in_delta = 0.0; action_taken = None
        if abs(current_portfolio_delta - TARGET_NET_DELTA) > HEDGE_THRESHOLD:
            hedge_amount_needed_in_delta = TARGET_NET_DELTA - current_portfolio_delta
            action_taken = 'buy_spot' if hedge_amount_needed_in_delta > 0 else 'sell_spot'
            hedge_pos_delta_val += hedge_amount_needed_in_delta
        if pd.isna(row['spot_hist']): continue 
        history.append({'timestamp': row['date_time'], 'option_delta_total_short_book': row['option_delta_total'], 'hedge_pos_delta_contribution': hedge_pos_delta_val, 'net_delta_after_hedge': row['option_delta_total'] + hedge_pos_delta_val, 'spot': row['spot_hist'], 'action': action_taken, 'hedge_trade_delta_impact': hedge_amount_needed_in_delta if action_taken else 0.0})
    df_history = pd.DataFrame(history)
    if df_history.empty: st.info("Pair sim history is empty after processing."); return
    if 'spot' not in df_history.columns or df_history['spot'].isna().all(): st.error("Spot data missing/invalid in history for plotting."); return
        
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Net Delta Components (Pair as Perps)", "Hedge Position (Delta Contribution) & Spot Price"), specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['option_delta_total_short_book'], name='Total Option Delta (Short Book)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['hedge_pos_delta_contribution'], name='Hedge Position Delta Contribution', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['net_delta_after_hedge'], name='Net Delta (Post-Hedge)'), row=1, col=1)
    fig.add_hline(y=TARGET_NET_DELTA, row=1,col=1, line_dash='solid', line_color='black', annotation_text="Target Net Delta")
    fig.add_hline(y=TARGET_NET_DELTA + HEDGE_THRESHOLD, row=1,col=1, line_dash='dot', line_color='red', annotation_text="Upper Threshold")
    fig.add_hline(y=TARGET_NET_DELTA - HEDGE_THRESHOLD, row=1,col=1, line_dash='dot', line_color='red', annotation_text="Lower Threshold")
    hedge_actions_plot_df = df_history[df_history['action'].notna()].copy()
    if not hedge_actions_plot_df.empty:
        buy_actions = hedge_actions_plot_df[hedge_actions_plot_df['action'] == 'buy_spot']
        sell_actions = hedge_actions_plot_df[hedge_actions_plot_df['action'] == 'sell_spot']
        if not buy_actions.empty: fig.add_trace(go.Scatter(x=buy_actions['timestamp'], y=buy_actions['net_delta_after_hedge'] - buy_actions['hedge_trade_delta_impact'], mode='markers', name='Buy Spot Hedge Trigger', marker=dict(symbol='triangle-up', size=8, color='lime'), hovertext=[f"Buy Spot (Δ Impact: {s:.3f})" for s in buy_actions['hedge_trade_delta_impact']]), row=1, col=1)
        if not sell_actions.empty: fig.add_trace(go.Scatter(x=sell_actions['timestamp'], y=sell_actions['net_delta_after_hedge'] - sell_actions['hedge_trade_delta_impact'], mode='markers', name='Sell Spot Hedge Trigger', marker=dict(symbol='triangle-down', size=8, color='red'), hovertext=[f"Sell Spot (Δ Impact: {s:.3f})" for s in sell_actions['hedge_trade_delta_impact']]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['hedge_pos_delta_contribution'], name='Hedge Position (Delta)'), row=2, col=1, secondary_y=False)
    if not df_history['spot'].isna().all(): fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['spot'], name='Spot Price', yaxis='y2'), row=2, col=1, secondary_y=True)
    fig.update_layout(height=700, legend_title_text='Metric', yaxis2=dict(overlaying='y', side='right', title='Spot Price'))
    fig.update_yaxes(title_text="Delta", row=1, col=1); fig.update_yaxes(title_text="Hedge Delta", row=2, col=1, secondary_y=False); fig.update_yaxes(title_text="Spot Price", row=2, col=1, secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    logging.info(f"{function_name}: Plotting complete for {current_call_instr} / {current_put_instr}")
# --- compute_delta1 (Unchanged) ---
def compute_delta1(row, S, snapshot_time_utc, funding_rate=0.0, is_perpetual=False):
    instr_name = row.get('instrument_name', 'N/A')
    option_type = row.get('option_type')
    k = row.get('k')
    sigma = row.get('iv_close')

    if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(option_type) or option_type not in ['C', 'P'] or pd.isna(funding_rate) or pd.isna(sigma):
        return np.nan

    if sigma < 1e-7:
        if option_type == 'C': return 1.0 if S > k else 0.0
        return -1.0 if S < k else 0.0
    
    T_eff = 0.0
    if is_perpetual:
        T_eff = 7.0 / 365.0 
    else: 
        expiry_date = row.get('expiry_datetime_col')
        if pd.isna(expiry_date): return np.nan
        if isinstance(expiry_date, pd.Timestamp): expiry_date = expiry_date.to_pydatetime()
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
        
        if not isinstance(snapshot_time_utc, dt.datetime): snapshot_time_utc = pd.to_datetime(snapshot_time_utc, utc=True).to_pydatetime()
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        
        T_eff = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)

    if T_eff < 1e-9: 
        if option_type == 'C': return 1.0 if S > k else 0.0
        return -1.0 if S < k else 0.0
        
    try:
        sigma_sqrt_T = sigma * math.sqrt(T_eff)
        if abs(sigma_sqrt_T) < 1e-12: return np.nan
        
        log_S_K = np.log(S / k)
        if not np.isfinite(log_S_K): return np.nan
        
        r_eff = funding_rate 
        
        d1_numerator = log_S_K + (r_eff + 0.5 * sigma**2) * T_eff
        d1 = d1_numerator / sigma_sqrt_T
        if not np.isfinite(d1): return np.nan
        
        delta_val = si.norm.cdf(d1) if option_type == 'C' else si.norm.cdf(d1) - 1.0
        if not np.isfinite(delta_val): return np.nan
        return delta_val
    except Exception as e:
        logging.error(f"Error in compute_delta1 for {instr_name} S={S} K={k} T_eff={T_eff} IV={sigma} FR={funding_rate}: {e}")
        return np.nan
# --- Hedging Classes (HedgeThalex, MatrixHedgeThalex, MatrixDeltaGammaHedgeSimple) and their plotters (Unchanged from previous correct version) ---
# ... (These classes are long, keeping them as they were in the previous corrected version)
# Ensure HedgeThalex.run_loop is the corrected one as provided in the previous step
class HedgeThalex: # Corrected run_loop
    def __init__(self, df_historical, spot_df, symbol="BTC", base_threshold=0.20, use_dynamic_threshold=False, iv_threshold_sensitivity=0.1, iv_low_ref=0.40, min_threshold=0.05, max_threshold=0.50, use_dynamic_hedge_size=False, min_hedge_ratio=0.50, max_hedge_ratio=1.0, iv_hedge_size_sensitivity=0.5, iv_high_ref=0.80, transaction_cost_bps=2):
        self.df_historical = df_historical.copy(); self.spot_df = spot_df.copy(); self.symbol = symbol.upper()
        self.base_threshold = abs(float(base_threshold)); self.use_dynamic_threshold = use_dynamic_threshold; self.iv_threshold_sensitivity = iv_threshold_sensitivity; self.iv_low_ref = iv_low_ref; self.min_threshold = abs(float(min_threshold)); self.max_threshold = abs(float(max_threshold)); self.use_dynamic_hedge_size = use_dynamic_hedge_size; self.min_hedge_ratio = max(0.0, min(1.0, float(min_hedge_ratio))); self.max_hedge_ratio = max(0.0, min(1.0, float(max_hedge_ratio))); self.iv_hedge_size_sensitivity = iv_hedge_size_sensitivity; self.iv_high_ref = iv_high_ref
        
        self.delta_history = [] 
        self.hedge_actions = [] 
        self.cumulative_hedge_delta = 0.0 
        self.transaction_cost_bps = transaction_cost_bps
        self.cumulative_pnl = 0.0
        self.cumulative_trading_costs = 0.0

        if self.symbol not in ["BTC", "ETH"]: raise ValueError("Incorrect symbol")
        req_hist_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'expiry_datetime_col', 'mark_price_close'] 
        if self.df_historical.empty or not all(c in self.df_historical.columns for c in req_hist_cols): raise ValueError(f"HedgeThalex: df_historical missing columns: {req_hist_cols}")
        if self.spot_df.empty or 'close' not in self.spot_df.columns or 'date_time' not in self.spot_df.columns: raise ValueError("HedgeThalex: spot_df must contain 'close' and 'date_time'")

    def _calculate_option_delta_at_ts(self, option_row, spot_price_at_ts, timestamp_at_ts, risk_free_rate):
        return compute_delta(option_row, spot_price_at_ts, timestamp_at_ts, risk_free_rate)

    def _get_current_mm_book_state(self, timestamp, spot_price): 
        try:
            df_at_ts = self.df_historical[self.df_historical['date_time'] == timestamp].copy()
            df_at_ts = df_at_ts[df_at_ts['expiry_datetime_col'] > timestamp] 
            if df_at_ts.empty: return 0.0, 0.0, 0.0, np.nan 
            risk_free_rate = st.session_state.get('risk_free_rate_input', 0.0)
            
            df_at_ts['delta_calc'] = df_at_ts.apply(lambda row: self._calculate_option_delta_at_ts(row, spot_price, timestamp, risk_free_rate), axis=1)
            
            valid_ivs = df_at_ts['iv_close'].dropna(); avg_iv = valid_ivs.mean() if not valid_ivs.empty else np.nan
            df_at_ts = df_at_ts.dropna(subset=['delta_calc', 'open_interest', 'mark_price_close']) 
            
            if df_at_ts.empty: return 0.0, 0.0, 0.0, avg_iv 
            
            mm_book_value = -(df_at_ts['mark_price_close'] * df_at_ts['open_interest']).sum(skipna=True)
            mm_book_delta = -(df_at_ts['delta_calc'] * df_at_ts['open_interest']).sum(skipna=True)
            mm_book_gamma = 0.0 

            return mm_book_value, mm_book_delta, mm_book_gamma, avg_iv
        except Exception as e:
            logging.error(f"Error in _get_current_mm_book_state (HedgeThalex) at {timestamp}: {e}", exc_info=True)
            return np.nan, np.nan, np.nan, np.nan

    def _delta_hedge_step(self, timestamp, spot_price, current_mm_book_delta, avg_iv):
        if pd.isna(current_mm_book_delta): return 0.0 
        current_net_delta_exposure = current_mm_book_delta + self.cumulative_hedge_delta
        current_threshold = self.base_threshold
        if self.use_dynamic_threshold and not pd.isna(avg_iv):
            dynamic_part = self.iv_threshold_sensitivity * (avg_iv - self.iv_low_ref)
            current_threshold = max(self.min_threshold, min(self.max_threshold, self.base_threshold + dynamic_part))
        
        trade_cost_this_step = 0.0; hedge_amount_delta_units = 0.0; sign = None
        
        if current_net_delta_exposure > current_threshold: 
            excess_delta = current_net_delta_exposure - current_threshold 
            hedge_amount_delta_units = -excess_delta 
            sign = 'sell'
        elif current_net_delta_exposure < -current_threshold: 
            shortfall_delta = current_net_delta_exposure - (-current_threshold) 
            hedge_amount_delta_units = -shortfall_delta
            sign = 'buy' 
        else: 
            return 0.0 
        
        actual_trade_size_underlying = hedge_amount_delta_units; current_hedge_ratio = 1.0
        if self.use_dynamic_hedge_size and not pd.isna(avg_iv):
            iv_range_for_ratio = max(1e-6, self.iv_high_ref - self.iv_low_ref)
            iv_pos_in_range = max(0, min(1, (avg_iv - self.iv_low_ref) / iv_range_for_ratio))
            target_ratio = self.min_hedge_ratio + (self.max_hedge_ratio - self.min_hedge_ratio) * iv_pos_in_range
            current_hedge_ratio = max(self.min_hedge_ratio, min(self.max_hedge_ratio, target_ratio))
            actual_trade_size_underlying *= current_hedge_ratio
        
        if abs(actual_trade_size_underlying) > 1e-9: 
            self.cumulative_hedge_delta += actual_trade_size_underlying 
            trade_notional = abs(actual_trade_size_underlying * spot_price) 
            trade_cost_this_step = trade_notional * (self.transaction_cost_bps / 10000.0)
            self.cumulative_trading_costs += trade_cost_this_step
            delta_after_hedge = current_mm_book_delta + self.cumulative_hedge_delta 
            
            action = {'timestamp': timestamp, 'action': sign, 
                      'size_underlying': abs(actual_trade_size_underlying), 
                      'spot_price_at_trade': spot_price, 
                      'delta_before_trade_in_step': current_net_delta_exposure, 
                      'delta_after_trade_in_step': delta_after_hedge, 
                      'threshold_used': current_threshold, 
                      'hedge_ratio_used': current_hedge_ratio, 
                      'avg_iv_at_ts': avg_iv, 
                      'trade_cost': trade_cost_this_step}
            self.hedge_actions.append(action)
        return trade_cost_this_step

    def run_loop(self, days=5):
        self.delta_history = []
        self.hedge_actions = []
        self.cumulative_hedge_delta = 0.0
        self.cumulative_pnl = 0.0
        self.cumulative_trading_costs = 0.0
        prev_portfolio_total_value_m2m = np.nan

        if self.df_historical.empty or self.spot_df.empty:
            logging.info("HedgeThalex.run_loop: df_historical or spot_df is empty. Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        latest_hist_ts = self.df_historical['date_time'].max()
        latest_spot_ts = self.spot_df['date_time'].max()
        if pd.isna(latest_hist_ts) or pd.isna(latest_spot_ts):
            logging.info(f"HedgeThalex.run_loop: Timestamps NaN. Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        latest_timestamp = min(latest_hist_ts, latest_spot_ts)
        min_data_ts_hist = self.df_historical['date_time'].min()
        min_data_ts_spot = self.spot_df['date_time'].min()

        if pd.isna(min_data_ts_hist) or pd.isna(min_data_ts_spot):
            logging.info(f"HedgeThalex.run_loop: Min data timestamp for hist or spot is NaN. Exiting.")
            return pd.DataFrame(), pd.DataFrame()
        min_data_ts = max(min_data_ts_hist, min_data_ts_spot)


        potential_start_timestamp = latest_timestamp - pd.Timedelta(days=days)
        start_timestamp = max(potential_start_timestamp, min_data_ts)

        if start_timestamp >= latest_timestamp:
            logging.info(f"HedgeThalex.run_loop: No valid simulation range ({start_timestamp} to {latest_timestamp}). Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        sim_options_df = self.df_historical[
            (self.df_historical['date_time'] >= start_timestamp) &
            (self.df_historical['date_time'] <= latest_timestamp)
        ].copy()
        
        sim_spot_df = self.spot_df[
            (self.spot_df['date_time'] >= start_timestamp) &
            (self.spot_df['date_time'] <= latest_timestamp)
        ].copy()

        if sim_options_df.empty or sim_spot_df.empty:
            logging.info("HedgeThalex.run_loop: No data after filtering for sim range. Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        loop_driving_timestamps = sorted(sim_options_df['date_time'].unique())
        if not loop_driving_timestamps:
            logging.info("HedgeThalex.run_loop: No unique driving timestamps. Exiting.")
            return pd.DataFrame(), pd.DataFrame()
        
        loop_timestamps_df = pd.DataFrame({'date_time': loop_driving_timestamps})
        spot_for_sim = pd.merge_asof(
            left=loop_timestamps_df.sort_values('date_time'),
            right=sim_spot_df[['date_time', 'close']].sort_values('date_time'),
            on='date_time', direction='nearest', tolerance=pd.Timedelta('10min') 
        )
        spot_for_sim['close'] = spot_for_sim['close'].ffill().bfill()
        spot_for_sim = spot_for_sim.dropna(subset=['date_time', 'close'])

        final_sim_timestamps = sorted(spot_for_sim['date_time'].unique())
        if not final_sim_timestamps:
            logging.info("HedgeThalex.run_loop: No final simulation timestamps after spot merge. Exiting.")
            return pd.DataFrame(), pd.DataFrame()

        for ts_idx, ts in enumerate(final_sim_timestamps):
            try:
                spot_price_at_ts_series = spot_for_sim.loc[spot_for_sim['date_time'] == ts, 'close']
                if spot_price_at_ts_series.empty: continue
                spot_price_at_ts = spot_price_at_ts_series.iloc[0]
                if pd.isna(spot_price_at_ts) or spot_price_at_ts <= 0: continue

                mm_book_value, mm_book_delta, _, avg_iv = self._get_current_mm_book_state(ts, spot_price_at_ts)
                
                if pd.isna(mm_book_delta) or pd.isna(mm_book_value):
                    self.delta_history.append({'timestamp': ts, 'spot_price': spot_price_at_ts, 'mm_book_delta': np.nan, 'cumulative_hedge_delta': self.cumulative_hedge_delta, 'net_portfolio_delta_before_step': np.nan, 'net_portfolio_delta_after_step': np.nan, 'threshold_used': np.nan, 'avg_iv_at_ts': avg_iv, 'pnl_step': 0.0, 'cumulative_pnl': self.cumulative_pnl, 'cumulative_trading_costs': self.cumulative_trading_costs})
                    prev_portfolio_total_value_m2m = np.nan 
                    continue
                
                net_portfolio_delta_before_step = mm_book_delta + self.cumulative_hedge_delta
                
                current_total_value_m2m = mm_book_value + (self.cumulative_hedge_delta * spot_price_at_ts)
                pnl_this_step_from_m2m = 0.0
                if ts_idx > 0 and not pd.isna(prev_portfolio_total_value_m2m):
                    pnl_this_step_from_m2m = current_total_value_m2m - prev_portfolio_total_value_m2m
                
                trade_cost_this_step = self._delta_hedge_step(ts, spot_price_at_ts, mm_book_delta, avg_iv)

                pnl_this_step_final = pnl_this_step_from_m2m - trade_cost_this_step
                self.cumulative_pnl += pnl_this_step_final
                
                net_portfolio_delta_after_step = mm_book_delta + self.cumulative_hedge_delta
                
                threshold_this_step = self.base_threshold
                if self.hedge_actions and self.hedge_actions[-1]['timestamp'] == ts:
                    threshold_this_step = self.hedge_actions[-1]['threshold_used']
                elif self.use_dynamic_threshold and not pd.isna(avg_iv):
                     dynamic_part = self.iv_threshold_sensitivity * (avg_iv - self.iv_low_ref)
                     threshold_this_step = max(self.min_threshold, min(self.max_threshold, self.base_threshold + dynamic_part))

                self.delta_history.append({
                    'timestamp': ts, 'spot_price': spot_price_at_ts,
                    'mm_book_delta': mm_book_delta,
                    'cumulative_hedge_delta': self.cumulative_hedge_delta,
                    'net_portfolio_delta_before_step': net_portfolio_delta_before_step,
                    'net_portfolio_delta_after_step': net_portfolio_delta_after_step,
                    'threshold_used': threshold_this_step,
                    'avg_iv_at_ts': avg_iv if pd.notna(avg_iv) else np.nan,
                    'pnl_step': pnl_this_step_final,
                    'cumulative_pnl': self.cumulative_pnl,
                    'cumulative_trading_costs': self.cumulative_trading_costs
                })
                
                prev_portfolio_total_value_m2m = mm_book_value + (self.cumulative_hedge_delta * spot_price_at_ts)

            except Exception as e_loop:
                logging.error(f"HedgeThalex.run_loop: Error in loop at {ts}: {e_loop}", exc_info=True)
                self.delta_history.append({'timestamp': ts, 'spot_price': np.nan, 'mm_book_delta': np.nan, 'cumulative_hedge_delta': self.cumulative_hedge_delta, 'net_portfolio_delta_before_step': np.nan, 'net_portfolio_delta_after_step': np.nan, 'threshold_used': np.nan, 'avg_iv_at_ts': np.nan, 'pnl_step': 0.0, 'cumulative_pnl': self.cumulative_pnl, 'cumulative_trading_costs': self.cumulative_trading_costs})
                prev_portfolio_total_value_m2m = np.nan

        logging.info(f"HedgeThalex.run_loop: Simulation finished. Delta history: {len(self.delta_history)}, Hedge actions: {len(self.hedge_actions)}")
        return pd.DataFrame(self.delta_history), pd.DataFrame(self.hedge_actions)
# All other classes and plotters (MatrixHedgeThalex, MatrixDeltaGammaHedgeSimple, plot_delta_hedging_thalex etc.)
# remain as they were in the previous fully corrected version.
# ...
class MatrixHedgeThalex: # Unchanged
    def __init__(self, df_historical, spot_df, symbol="BTC", transaction_cost_bps=2):
        self.df_historical = df_historical.copy(); self.spot_df = spot_df.copy(); self.symbol = symbol.upper()
        self.portfolio_state = []; self.hedge_actions = []; self.current_hedge_n1 = 0.0; self.current_B = 0.0
        self.transaction_cost_bps = transaction_cost_bps
        self.cumulative_pnl = 0.0; self.cumulative_trading_costs = 0.0
        req_hist_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'mark_price_close', 'expiry_datetime_col']
        if self.df_historical.empty or not all(c in self.df_historical.columns for c in req_hist_cols): raise ValueError(f"MatrixHedgeThalex: df_historical missing columns")
        if self.spot_df.empty or 'close' not in self.spot_df.columns or 'date_time' not in self.spot_df.columns: raise ValueError("MatrixHedgeThalex: spot_df must contain 'close' and 'date_time'")

    def _get_mm_portfolio_value_delta(self, timestamp, spot_price):
        try:
            df_at_ts = self.df_historical[self.df_historical['date_time'] == timestamp].copy()
            df_at_ts = df_at_ts[df_at_ts['expiry_datetime_col'] > timestamp]
            if df_at_ts.empty: return 0.0, 0.0
            risk_free_rate = st.session_state.get('risk_free_rate_input', 0.0)
            df_at_ts['option_value_abs'] = df_at_ts['mark_price_close'] * df_at_ts['open_interest']
            df_at_ts['option_delta_abs'] = df_at_ts.apply(lambda row: compute_delta(row, spot_price, timestamp, risk_free_rate), axis=1) * df_at_ts['open_interest']
            mm_option_book_value = -df_at_ts['option_value_abs'].sum(skipna=True)
            mm_option_book_delta = -df_at_ts['option_delta_abs'].sum(skipna=True)
            return mm_option_book_value, mm_option_book_delta
        except Exception as e: logging.error(f"Error in _get_mm_portfolio_value_delta (MatrixHedgeThalex) at {timestamp}: {e}"); return np.nan, np.nan

    def _solve_hedge_system(self, spot_price, mm_book_value, mm_book_delta):
        if pd.isna(spot_price) or pd.isna(mm_book_value) or pd.isna(mm_book_delta): return np.nan, np.nan
        try:
            target_n1 = -mm_book_delta 
            target_B = -mm_book_value - (target_n1 * spot_price) 
            return target_B, target_n1
        except Exception as e: logging.error(f"Error solving matrix hedge system (MatrixHedgeThalex): {e}"); return np.nan, np.nan

    def run_loop(self, days=5):
        if self.df_historical.empty or self.spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        latest_hist_ts = self.df_historical['date_time'].max(); latest_spot_ts = self.spot_df['date_time'].max()
        if pd.isna(latest_hist_ts) or pd.isna(latest_spot_ts): return pd.DataFrame(), pd.DataFrame()
        latest_timestamp = min(latest_hist_ts, latest_spot_ts);
        start_timestamp_calc = latest_timestamp - pd.Timedelta(days=min(days, 30)) # Increased default sim range slightly for matrix
        
        sim_options_df = self.df_historical[(self.df_historical['date_time'] >= start_timestamp_calc) & (self.df_historical['date_time'] <= latest_timestamp)].copy()
        sim_spot_df = self.spot_df[(self.spot_df['date_time'] >= start_timestamp_calc) & (self.spot_df['date_time'] <= latest_timestamp)].copy()

        if sim_options_df.empty or sim_spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        
        effective_start_time = max(sim_options_df['date_time'].min(), sim_spot_df['date_time'].min())
        if pd.isna(effective_start_time) or effective_start_time > latest_timestamp : return pd.DataFrame(), pd.DataFrame()
        
        loop_timestamps = sorted(sim_options_df['date_time'].unique())
        if not loop_timestamps: return pd.DataFrame(), pd.DataFrame()
        
        loop_timestamps_df = pd.DataFrame({'date_time': loop_timestamps})
        spot_for_sim = pd.merge_asof(left=loop_timestamps_df.sort_values('date_time'), right=sim_spot_df[['date_time', 'close']].sort_values('date_time'), on='date_time', direction='nearest', tolerance=pd.Timedelta('10min'))
        spot_for_sim['close'] = spot_for_sim['close'].ffill().bfill(); spot_for_sim = spot_for_sim.dropna(subset=['date_time', 'close'])
        
        final_timestamps = sorted(spot_for_sim['date_time'].unique())
        if not final_timestamps: return pd.DataFrame(), pd.DataFrame()
        
        self.portfolio_state = []; self.hedge_actions = []
        self.current_hedge_n1 = 0.0; self.current_B = 0.0
        self.cumulative_pnl = 0.0; self.cumulative_trading_costs = 0.0
        trade_tolerance_n1 = 1e-3 
        prev_portfolio_total_value_m2m = np.nan

        for ts_idx, ts in enumerate(final_timestamps):
            try:
                spot_price_at_ts = spot_for_sim.loc[spot_for_sim['date_time'] == ts, 'close'].iloc[0]
                if pd.isna(spot_price_at_ts) or spot_price_at_ts <= 0: continue
                
                mm_book_value_at_ts, mm_book_delta_at_ts = self._get_mm_portfolio_value_delta(timestamp=ts, spot_price=spot_price_at_ts)
                if pd.isna(mm_book_value_at_ts) or pd.isna(mm_book_delta_at_ts): continue
                
                current_portfolio_total_value_m2m = mm_book_value_at_ts + self.current_hedge_n1 * spot_price_at_ts + self.current_B
                pnl_this_step = 0.0
                if ts_idx > 0 and not pd.isna(prev_portfolio_total_value_m2m): 
                    pnl_this_step = current_portfolio_total_value_m2m - prev_portfolio_total_value_m2m
                
                target_B_val, target_n1_val = self._solve_hedge_system(spot_price_at_ts, mm_book_value_at_ts, mm_book_delta_at_ts)
                trade_cost_this_step = 0.0
                
                if pd.notna(target_n1_val):
                    trade_size_n1 = target_n1_val - self.current_hedge_n1
                    if abs(trade_size_n1) > trade_tolerance_n1:
                        action_n1 = 'buy' if trade_size_n1 > 0 else 'sell'
                        trade_notional = abs(trade_size_n1 * spot_price_at_ts)
                        trade_cost_this_step = trade_notional * (self.transaction_cost_bps / 10000.0)
                        self.cumulative_trading_costs += trade_cost_this_step
                        
                        self.hedge_actions.append({'timestamp': ts, 'action': action_n1, 'instrument_type': 'underlying', 'size': abs(trade_size_n1), 'target_qty': target_n1_val, 'qty_before': self.current_hedge_n1, 'spot_price': spot_price_at_ts, 'trade_cost': trade_cost_this_step})
                        
                        self.current_B -= (trade_size_n1 * spot_price_at_ts) 
                        self.current_B -= trade_cost_this_step 
                        self.current_hedge_n1 = target_n1_val 
                
                pnl_this_step -= trade_cost_this_step 
                self.cumulative_pnl += pnl_this_step
                
                if pd.notna(target_B_val): self.current_B = target_B_val 
                
                net_delta_final = mm_book_delta_at_ts + self.current_hedge_n1
                net_value_final = mm_book_value_at_ts + self.current_hedge_n1 * spot_price_at_ts + self.current_B
                prev_portfolio_total_value_m2m = net_value_final 
                
                self.portfolio_state.append({'timestamp': ts, 'spot_price': spot_price_at_ts, 'mm_book_value': mm_book_value_at_ts, 'mm_book_delta': mm_book_delta_at_ts, 'target_n1': target_n1_val, 'target_B': target_B_val, 'current_n1': self.current_hedge_n1, 'current_B': self.current_B, 'net_delta_final': net_delta_final, 'net_value_final': net_value_final, 'pnl_step': pnl_this_step, 'cumulative_pnl': self.cumulative_pnl, 'cumulative_trading_costs': self.cumulative_trading_costs})
            except Exception as e: logging.error(f"Error in MatrixHedgeThalex run_loop at {ts}: {e}", exc_info=True)
        return pd.DataFrame(self.portfolio_state), pd.DataFrame(self.hedge_actions)

class MatrixDeltaGammaHedgeSimple: # Unchanged
    def __init__(self, df_portfolio_options, spot_df, symbol="BTC", risk_free_rate=0.0, gamma_hedge_instrument_details=None, transaction_cost_bps_spot=2, transaction_cost_bps_option=5):
        self.df_portfolio_options = df_portfolio_options.copy(); self.spot_df = spot_df.copy(); self.symbol = symbol.upper(); self.risk_free_rate = risk_free_rate; self.gamma_hedge_instrument_details = gamma_hedge_instrument_details
        self.portfolio_state_log = []; self.hedge_actions_log = []
        self.current_underlying_hedge_qty = 0.0; self.current_gamma_option_hedge_qty = 0.0; self.current_B_cash = 0.0
        self.transaction_cost_bps_spot = transaction_cost_bps_spot; self.transaction_cost_bps_option = transaction_cost_bps_option
        self.cumulative_pnl = 0.0; self.cumulative_trading_costs = 0.0
        self._validate_inputs()

    def _validate_inputs(self): 
        if self.symbol not in ["BTC", "ETH"]: raise ValueError(f"Incorrect symbol: {self.symbol}")
        req_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'mark_price_close', 'expiry_datetime_col']
        if self.df_portfolio_options.empty or not all(c in self.df_portfolio_options.columns for c in req_cols):
            missing = [c for c in req_cols if c not in self.df_portfolio_options.columns]; raise ValueError(f"MatrixDeltaGammaHedge: df_portfolio_options missing columns: {missing}")
        for df_check in [self.df_portfolio_options, self.spot_df]: # TZ handling
            if 'date_time' not in df_check.columns: continue 
            if not pd.api.types.is_datetime64_any_dtype(df_check['date_time']): df_check['date_time'] = pd.to_datetime(df_check['date_time'], utc=True)
            elif df_check['date_time'].dt.tz is None: df_check['date_time'] = df_check['date_time'].dt.tz_localize('UTC')
            elif df_check['date_time'].dt.tz != dt.timezone.utc: df_check['date_time'] = df_check['date_time'].dt.tz_convert('UTC')
        
        if 'expiry_datetime_col' in self.df_portfolio_options.columns: 
            exp_col = self.df_portfolio_options['expiry_datetime_col'] 
            if not pd.api.types.is_datetime64_any_dtype(exp_col): self.df_portfolio_options['expiry_datetime_col'] = pd.to_datetime(exp_col, utc=True)
            elif exp_col.dt.tz is None: self.df_portfolio_options['expiry_datetime_col'] = exp_col.dt.tz_localize('UTC')
            elif exp_col.dt.tz != dt.timezone.utc: self.df_portfolio_options['expiry_datetime_col'] = exp_col.dt.tz_convert('UTC')
        
        if self.gamma_hedge_instrument_details: 
            hedger_exp = self.gamma_hedge_instrument_details.get('expiry_datetime_col')
            if hedger_exp and not isinstance(hedger_exp, dt.datetime):
                try: self.gamma_hedge_instrument_details['expiry_datetime_col'] = pd.to_datetime(hedger_exp, utc=True).to_pydatetime()
                except Exception as e: raise ValueError(f"Gamma hedger expiry_datetime_col invalid: {hedger_exp} - Error: {e}")

    def _get_mm_portfolio_greeks(self, timestamp, spot_price): 
        if self.df_portfolio_options.empty: return 0.0, 0.0, 0.0
        options_at_ts = self.df_portfolio_options[self.df_portfolio_options['date_time'] == timestamp].copy()
        if 'expiry_datetime_col' not in options_at_ts.columns or options_at_ts['expiry_datetime_col'].isnull().all() :
             logging.warning(f"Timestamp {timestamp}: 'expiry_datetime_col' missing or all null in options_at_ts. Cannot filter expired options.")
        else:
            options_at_ts = options_at_ts[options_at_ts['expiry_datetime_col'] > timestamp]
        
        if options_at_ts.empty: return 0.0, 0.0, 0.0
        
        req_greek_cols = ['mark_price_close', 'open_interest', 'k', 'iv_close', 'option_type', 'expiry_datetime_col']
        if not all(col in options_at_ts.columns for col in req_greek_cols):
            logging.error(f"Timestamp {timestamp}: Missing one or more required columns for greek calculation in options_at_ts: {options_at_ts.columns.tolist()}")
            return np.nan, np.nan, np.nan

        options_at_ts['value_abs'] = options_at_ts['mark_price_close'] * options_at_ts['open_interest']
        options_at_ts['delta_abs'] = options_at_ts.apply(lambda r: compute_delta(r, spot_price, timestamp, self.risk_free_rate), axis=1) * options_at_ts['open_interest']
        options_at_ts['gamma_abs'] = options_at_ts.apply(lambda r: compute_gamma(r, spot_price, timestamp, self.risk_free_rate), axis=1) * options_at_ts['open_interest']
        
        mm_value = -options_at_ts['value_abs'].sum(skipna=True)
        mm_delta = -options_at_ts['delta_abs'].sum(skipna=True)
        mm_gamma = -options_at_ts['gamma_abs'].sum(skipna=True)
        return mm_value, mm_delta, mm_gamma

    def _get_gamma_hedger_greeks_and_price(self, timestamp, spot_price):
        if not self.gamma_hedge_instrument_details:
            return np.nan, np.nan, np.nan

        details = self.gamma_hedge_instrument_details
        hedger_name = details['name'] 

        hedger_data_at_ts = self.df_portfolio_options[
            (self.df_portfolio_options['instrument_name'] == hedger_name) &
            (self.df_portfolio_options['date_time'] == timestamp)
        ]

        if hedger_data_at_ts.empty:
            logging.warning(f"Hedger {hedger_name} data not found at timestamp {timestamp}")
            return np.nan, np.nan, np.nan

        hedger_row_series = hedger_data_at_ts.iloc[0].copy() 

        current_hedger_iv = hedger_row_series.get('iv_close', np.nan)
        current_hedger_mark_price = hedger_row_series.get('mark_price_close', np.nan)

        if pd.isna(current_hedger_iv) or current_hedger_iv <= 0:
            logging.warning(f"Hedger {hedger_name} IV invalid at {timestamp}: {current_hedger_iv}")
            return np.nan, np.nan, current_hedger_mark_price

        hedger_delta = compute_delta(hedger_row_series, spot_price, timestamp, self.risk_free_rate)
        hedger_gamma = compute_gamma(hedger_row_series, spot_price, timestamp, self.risk_free_rate)
        
        MIN_GAMMA_FOR_GREEK_VALIDITY = 1e-7 
        if pd.isna(hedger_delta) or pd.isna(hedger_gamma) or abs(hedger_gamma) < MIN_GAMMA_FOR_GREEK_VALIDITY:
            logging.warning(f"Hedger {hedger_name} greeks invalid or gamma too small at {timestamp}. D={hedger_delta}, G={hedger_gamma}")
            return np.nan, np.nan, current_hedger_mark_price

        return hedger_delta, hedger_gamma, current_hedger_mark_price

    def _solve_delta_gamma_hedge_system(self, mm_portfolio_value, mm_portfolio_delta, mm_portfolio_gamma, spot_price, hedger_delta, hedger_gamma, hedger_price):
        if pd.isna(mm_portfolio_delta) or pd.isna(mm_portfolio_gamma) or pd.isna(spot_price): return np.nan, np.nan, np.nan
        n_g_val = 0.0
        if pd.notna(hedger_gamma) and abs(hedger_gamma) > 1e-9: n_g_val = -mm_portfolio_gamma / hedger_gamma
        else: 
            n_g_val = 0.0 
            
        n_u_val = -(mm_portfolio_delta + n_g_val * (hedger_delta if pd.notna(hedger_delta) else 0.0))
        target_B_val = -(mm_portfolio_value + n_u_val * spot_price + n_g_val * (hedger_price if pd.notna(hedger_price) and n_g_val != 0.0 else 0.0))
        
        return target_B_val, n_u_val, n_g_val

    def run_loop(self, days=5):
        if self.df_portfolio_options.empty or self.spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        latest_hist_ts = self.df_portfolio_options['date_time'].max(); latest_spot_ts = self.spot_df['date_time'].max()
        if pd.isna(latest_hist_ts) or pd.isna(latest_spot_ts): return pd.DataFrame(), pd.DataFrame()
        latest_timestamp = min(latest_hist_ts, latest_spot_ts)
        
        min_data_ts_hist = self.df_portfolio_options['date_time'].min()
        min_data_ts_spot = self.spot_df['date_time'].min()
        if pd.isna(min_data_ts_hist) or pd.isna(min_data_ts_spot): return pd.DataFrame(), pd.DataFrame()
        min_data_ts = max(min_data_ts_hist, min_data_ts_spot)

        potential_start_timestamp = latest_timestamp - pd.Timedelta(days=days); start_timestamp = max(potential_start_timestamp, min_data_ts)
        if start_timestamp >= latest_timestamp : return pd.DataFrame(), pd.DataFrame()
        
        sim_options_df = self.df_portfolio_options[(self.df_portfolio_options['date_time'] >= start_timestamp) & (self.df_portfolio_options['date_time'] <= latest_timestamp)].copy()
        sim_spot_df = self.spot_df[(self.spot_df['date_time'] >= start_timestamp) & (self.spot_df['date_time'] <= latest_timestamp)].copy()
        if sim_options_df.empty or sim_spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        
        loop_driving_timestamps = sorted(sim_options_df['date_time'].unique())
        if not loop_driving_timestamps: return pd.DataFrame(), pd.DataFrame()
        
        loop_timestamps_df = pd.DataFrame({'date_time': loop_driving_timestamps})
        spot_for_sim = pd.merge_asof(left=loop_timestamps_df.sort_values('date_time'), right=sim_spot_df[['date_time', 'close']].sort_values('date_time'), on='date_time', direction='nearest', tolerance=pd.Timedelta('10min'))
        spot_for_sim['close'] = spot_for_sim['close'].ffill().bfill(); spot_for_sim = spot_for_sim.dropna(subset=['date_time', 'close'])
        if spot_for_sim.empty: return pd.DataFrame(), pd.DataFrame()
        
        final_sim_timestamps = sorted(spot_for_sim['date_time'].unique())
        if not final_sim_timestamps: return pd.DataFrame(), pd.DataFrame()
        
        self.portfolio_state_log = []; self.hedge_actions_log = []
        self.current_underlying_hedge_qty = 0.0; self.current_gamma_option_hedge_qty = 0.0; self.current_B_cash = 0.0
        self.cumulative_pnl = 0.0; self.cumulative_trading_costs = 0.0
        trade_tolerance_underlying = 1e-3; trade_tolerance_option = 1e-2 
        prev_portfolio_total_value_m2m = np.nan

        for ts_idx, ts in enumerate(final_sim_timestamps):
            try:
                spot_price_at_ts = spot_for_sim.loc[spot_for_sim['date_time'] == ts, 'close'].iloc[0]
                if pd.isna(spot_price_at_ts) or spot_price_at_ts <= 0: continue
                
                mm_val, mm_delta, mm_gamma = self._get_mm_portfolio_greeks(ts, spot_price_at_ts)
                if pd.isna(mm_delta) or pd.isna(mm_gamma) or pd.isna(mm_val): continue
                
                hedger_P_for_m2m = 0.0 
                if self.gamma_hedge_instrument_details and self.current_gamma_option_hedge_qty != 0:
                    _, _, hedger_P_for_m2m_calc = self._get_gamma_hedger_greeks_and_price(ts, spot_price_at_ts)
                    if pd.notna(hedger_P_for_m2m_calc): hedger_P_for_m2m = hedger_P_for_m2m_calc
                
                current_portfolio_total_value_m2m = mm_val + self.current_underlying_hedge_qty * spot_price_at_ts + self.current_gamma_option_hedge_qty * hedger_P_for_m2m + self.current_B_cash
                pnl_this_step = 0.0
                if ts_idx > 0 and not pd.isna(prev_portfolio_total_value_m2m): 
                    pnl_this_step = current_portfolio_total_value_m2m - prev_portfolio_total_value_m2m
                
                hedger_D, hedger_G, hedger_P_trade = np.nan, np.nan, np.nan 
                if self.gamma_hedge_instrument_details: 
                    hedger_D, hedger_G, hedger_P_trade = self._get_gamma_hedger_greeks_and_price(ts, spot_price_at_ts)
                
                target_B, target_n_u, target_n_g = self._solve_delta_gamma_hedge_system(mm_val, mm_delta, mm_gamma, spot_price_at_ts, hedger_D, hedger_G, hedger_P_trade)
                trade_cost_this_step = 0.0
                
                if pd.notna(target_n_u):
                    trade_size_u = target_n_u - self.current_underlying_hedge_qty
                    if abs(trade_size_u) > trade_tolerance_underlying:
                        action_u = 'buy' if trade_size_u > 0 else 'sell'; cost_u_trade_val = trade_size_u * spot_price_at_ts 
                        cost_u_tc = abs(cost_u_trade_val) * (self.transaction_cost_bps_spot / 10000.0); trade_cost_this_step += cost_u_tc
                        self.current_B_cash -= (cost_u_trade_val + cost_u_tc) 
                        self.hedge_actions_log.append({'timestamp': ts, 'instrument': self.symbol + '-SPOT', 'action': action_u, 'size': abs(trade_size_u), 'price': spot_price_at_ts, 'type': 'delta_underlying', 'cost': cost_u_tc})
                        self.current_underlying_hedge_qty = target_n_u
                
                if self.gamma_hedge_instrument_details and pd.notna(target_n_g) and pd.notna(hedger_P_trade) and abs(hedger_G if pd.notna(hedger_G) else 0) > 1e-9 : 
                    trade_size_g = target_n_g - self.current_gamma_option_hedge_qty
                    if abs(trade_size_g) > trade_tolerance_option:
                        action_g = 'buy' if trade_size_g > 0 else 'sell'; cost_g_trade_val = trade_size_g * hedger_P_trade
                        cost_g_tc = abs(cost_g_trade_val) * (self.transaction_cost_bps_option / 10000.0); trade_cost_this_step += cost_g_tc
                        self.current_B_cash -= (cost_g_trade_val + cost_g_tc)
                        self.hedge_actions_log.append({'timestamp': ts, 'instrument': self.gamma_hedge_instrument_details['name'], 'action': action_g, 'size': abs(trade_size_g), 'price': hedger_P_trade, 'type': 'gamma_option', 'cost': cost_g_tc})
                        self.current_gamma_option_hedge_qty = target_n_g
                
                pnl_this_step -= trade_cost_this_step
                self.cumulative_pnl += pnl_this_step; self.cumulative_trading_costs += trade_cost_this_step
                
                if pd.notna(target_B) : self.current_B_cash = target_B 

                final_net_delta = mm_delta + self.current_underlying_hedge_qty 
                final_net_gamma = mm_gamma
                if self.current_gamma_option_hedge_qty != 0 and pd.notna(hedger_D) and pd.notna(hedger_G): 
                    final_net_delta += self.current_gamma_option_hedge_qty * hedger_D
                    final_net_gamma += self.current_gamma_option_hedge_qty * hedger_G
                
                final_net_value = mm_val + self.current_underlying_hedge_qty * spot_price_at_ts 
                if self.current_gamma_option_hedge_qty != 0 and pd.notna(hedger_P_trade): 
                    final_net_value += self.current_gamma_option_hedge_qty * hedger_P_trade
                final_net_value += self.current_B_cash 

                prev_portfolio_total_value_m2m = final_net_value
                
                self.portfolio_state_log.append({'timestamp': ts, 'spot_price': spot_price_at_ts, 'mm_portfolio_value': mm_val, 'mm_portfolio_delta': mm_delta, 'mm_portfolio_gamma': mm_gamma, 'target_B': target_B, 'target_n_underlying': target_n_u, 'target_n_gamma_opt': target_n_g, 'current_n_underlying': self.current_underlying_hedge_qty, 'current_n_gamma_opt': self.current_gamma_option_hedge_qty, 'current_B_cash': self.current_B_cash, 'hedger_delta_at_ts': hedger_D, 'hedger_gamma_at_ts': hedger_G, 'hedger_price_at_ts': hedger_P_trade, 'net_delta_final': final_net_delta, 'net_gamma_final': final_net_gamma, 'net_value_final': final_net_value, 'pnl_step': pnl_this_step, 'cumulative_pnl': self.cumulative_pnl, 'cumulative_trading_costs': self.cumulative_trading_costs})
            except Exception as e: logging.error(f"Error in MatrixDeltaGammaHedgeSimple run_loop at {ts}: {e}", exc_info=True)
        return pd.DataFrame(self.portfolio_state_log), pd.DataFrame(self.hedge_actions_log)

def plot_delta_hedging_thalex(delta_df, hedge_df, base_threshold, use_dynamic_threshold, symbol, spot_price_latest, spot_df_hist): # Unchanged
    st.subheader(f"Traditional Delta Hedging Simulation ({symbol})")
    dynamic_info = "Dynamic Threshold" if use_dynamic_threshold else f"Fixed Threshold ({base_threshold:.2f})"
    st.caption(f"Strategy: Neutralize delta of a hypothetical *short* option book. Plot shows MM's Net Delta. Threshold: {dynamic_info}")
    if delta_df.empty: st.info("No data for Traditional Delta Hedging plot."); return
    try:
        delta_df_local = delta_df.copy(); hedge_df_local = hedge_df.copy() if not hedge_df.empty else pd.DataFrame(); spot_df_local_orig = spot_df_hist.copy()
        for df_to_conv in [delta_df_local, hedge_df_local, spot_df_local_orig]: 
            if 'timestamp' in df_to_conv.columns: df_to_conv['timestamp'] = pd.to_datetime(df_to_conv['timestamp'])
            if 'date_time' in df_to_conv.columns: df_to_conv['date_time'] = pd.to_datetime(df_to_conv['date_time'])
        delta_df_local = delta_df_local.sort_values('timestamp')
        if not hedge_df_local.empty: hedge_df_local = hedge_df_local.sort_values('timestamp')
        if 'date_time' in spot_df_local_orig.columns: spot_df_local_orig = spot_df_local_orig.sort_values('date_time')
    
    except Exception as e: st.error(f"Error preparing data for delta hedge plot: {e}"); return
    
    delta_col_to_plot = 'net_portfolio_delta_after_step' 
    if delta_col_to_plot not in delta_df_local.columns : 
        st.error(f"Required delta column '{delta_col_to_plot}' not found in delta_df for plotting."); return
            
    valid_delta_df_plot = delta_df_local.dropna(subset=[delta_col_to_plot, 'timestamp']).copy()
    if valid_delta_df_plot.empty: st.info("No valid delta data to plot after NaN drop."); return
    
    latest_delta_val = valid_delta_df_plot[delta_col_to_plot].iloc[-1]; latest_ts_val = valid_delta_df_plot['timestamp'].iloc[-1]
    plot_start_time = valid_delta_df_plot['timestamp'].min(); plot_end_time = valid_delta_df_plot['timestamp'].max()
    
    spot_df_plot = pd.DataFrame()
    if 'date_time' in spot_df_local_orig.columns and 'close' in spot_df_local_orig.columns:
        spot_for_merge = spot_df_local_orig[['date_time', 'close']].rename(columns={'date_time':'timestamp'})
        spot_df_plot = pd.merge_asof(
            valid_delta_df_plot[['timestamp']].sort_values('timestamp'),
            spot_for_merge.sort_values('timestamp'),
            on='timestamp', direction='nearest', tolerance=pd.Timedelta('10min')
        ).dropna(subset=['close'])

    hedge_df_plot = hedge_df_local[(hedge_df_local['timestamp'] >= plot_start_time) & (hedge_df_local['timestamp'] <= plot_end_time)].copy() if not hedge_df_local.empty else pd.DataFrame()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Net Portfolio Delta & Spot Price", "Cumulative P&L and Trading Costs"), specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=valid_delta_df_plot['timestamp'], y=valid_delta_df_plot[delta_col_to_plot], name='Net Portfolio Delta', line=dict(color='blue')), row=1, col=1, secondary_y=False)
    
    if not spot_df_plot.empty: fig.add_trace(go.Scatter(x=spot_df_plot['timestamp'], y=spot_df_plot['close'], name='Spot Price', line=dict(color='orange')), row=1, col=1, secondary_y=True)
    
    delta_before_col = 'delta_before_trade_in_step' 
    size_col = 'size_underlying'
    
    if not hedge_df_plot.empty and delta_before_col in hedge_df_plot.columns and size_col in hedge_df_plot.columns :
        triggers = hedge_df_plot.dropna(subset=[delta_before_col, size_col])
        buys = triggers[triggers['action'] == 'buy']; sells = triggers[triggers['action'] == 'sell']
        if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys[delta_before_col], mode='markers', name='Buy Hedge Triggered', marker=dict(symbol='triangle-up', size=10, color='green'), hoverinfo='text', text=[f"Buy {s:.2f}" for s in buys[size_col]]), row=1, col=1, secondary_y=False)
        if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells[delta_before_col], mode='markers', name='Sell Hedge Triggered', marker=dict(symbol='triangle-down', size=10, color='red'), hoverinfo='text', text=[f"Sell {s:.2f}" for s in sells[size_col]]), row=1, col=1, secondary_y=False)
    
    max_dyn_thresh_val, min_dyn_thresh_val = base_threshold, -base_threshold
    if use_dynamic_threshold and 'threshold_used' in valid_delta_df_plot.columns and valid_delta_df_plot['threshold_used'].notna().any():
        thresh_df = valid_delta_df_plot.dropna(subset=['threshold_used'])
        if not thresh_df.empty:
            fig.add_trace(go.Scatter(x=thresh_df['timestamp'], y=thresh_df['threshold_used'], mode='lines', name='Upper Threshold', line=dict(color='red', width=1, dash='dash'), connectgaps=False), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=thresh_df['timestamp'], y=-thresh_df['threshold_used'], mode='lines', name='Lower Threshold', line=dict(color='red', width=1, dash='dash'), connectgaps=False), row=1, col=1, secondary_y=False)
            if thresh_df['threshold_used'].notna().any(): max_dyn_thresh_val = thresh_df['threshold_used'].max(); min_dyn_thresh_val = -max_dyn_thresh_val 
    else: 
        fig.add_hline(y=base_threshold, line_dash="dash", line_color="red", row=1, col=1, secondary_y=False, annotation_text=f"Thresh: {base_threshold:.2f}")
        fig.add_hline(y=-base_threshold, line_dash="dash", line_color="red", row=1, col=1, secondary_y=False, annotation_text=f"Thresh: {-base_threshold:.2f}")

    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=1, secondary_y=False, annotation_text="Neutral")
    
    if 'cumulative_pnl' in valid_delta_df_plot.columns: fig.add_trace(go.Scatter(x=valid_delta_df_plot['timestamp'], y=valid_delta_df_plot['cumulative_pnl'], name='Cumulative P&L (Net)', line=dict(color='green')), row=2, col=1, secondary_y=False)
    if 'cumulative_trading_costs' in valid_delta_df_plot.columns: fig.add_trace(go.Scatter(x=valid_delta_df_plot['timestamp'], y=valid_delta_df_plot['cumulative_trading_costs'], name='Cumulative Trading Costs', line=dict(color='red', dash='dot')), row=2, col=1, secondary_y=True)
    
    min_spot_r, max_spot_r = (spot_price_latest or 0) * 0.95, (spot_price_latest or 1) * 1.05
    if not spot_df_plot.empty and spot_df_plot['close'].notna().any():
        min_s, max_s = spot_df_plot['close'].min(), spot_df_plot['close'].max()
        if pd.notna(min_s) and pd.notna(max_s): min_spot_r, max_spot_r = min_s * 0.99, max_s * 1.01
    
    y_min_delta_plot = min(min_dyn_thresh_val, valid_delta_df_plot[delta_col_to_plot].min() if valid_delta_df_plot[delta_col_to_plot].notna().any() else min_dyn_thresh_val)
    y_max_delta_plot = max(max_dyn_thresh_val, valid_delta_df_plot[delta_col_to_plot].max() if valid_delta_df_plot[delta_col_to_plot].notna().any() else max_dyn_thresh_val)
    pad_delta = (y_max_delta_plot - y_min_delta_plot) * 0.1 if (y_max_delta_plot - y_min_delta_plot) > 1e-6 else 0.1
    
    fig.update_layout(title=f"MM Net Portfolio Delta ({symbol}) | Spot: ${spot_price_latest:,.2f}" if pd.notna(spot_price_latest) else f"MM Net Portfolio Delta ({symbol})", xaxis_title="Timestamp (UTC)", 
                      yaxis=dict(title="Net Delta", range=[y_min_delta_plot - pad_delta, y_max_delta_plot + pad_delta]), 
                      yaxis2=dict(title="Spot Price (USD)", overlaying='y', side='right', range=[min_spot_r, max_spot_r], showgrid=False) if not spot_df_plot.empty else None, 
                      height=700, legend_title_text='Metrics', hovermode='x unified')
    fig.update_yaxes(title_text="Delta", row=1, col=1, secondary_y=False); fig.update_yaxes(title_text="Spot Price", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1, secondary_y=False); fig.update_yaxes(title_text="Costs ($)", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(range=[plot_start_time, plot_end_time], tickformat="%Y-%m-%d\n%H:%M", row=2, col=1) 
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"Latest Sim Net Delta (at {latest_ts_val.strftime('%Y-%m-%d %H:%M')}): {latest_delta_val:.4f}")
    if 'cumulative_pnl' in valid_delta_df_plot and not valid_delta_df_plot['cumulative_pnl'].empty: st.write(f"Final Cumulative P&L: ${valid_delta_df_plot['cumulative_pnl'].iloc[-1]:,.2f}")
    if 'cumulative_trading_costs' in valid_delta_df_plot and not valid_delta_df_plot['cumulative_trading_costs'].empty: st.write(f"Total Trading Costs: ${valid_delta_df_plot['cumulative_trading_costs'].iloc[-1]:,.2f}")

def plot_matrix_hedge_thalex(portfolio_state_df, hedge_actions_df, symbol): # Unchanged
    st.subheader(f"Matrix Delta Hedging Simulation ({symbol} - MM Short Book)")
    if portfolio_state_df.empty: st.info("No data for Matrix Delta Hedging plot."); return
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, specs=[[{"secondary_y": False}],[{"secondary_y": True}],[{"secondary_y": True}]], subplot_titles=("Net Portfolio Delta & Components", f"Hedge Position ({symbol}) & Spot Price", "Cumulative P&L and Costs"))
    if 'net_delta_final' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['net_delta_final'], name='Net Portfolio Delta'), row=1, col=1)
    if 'mm_book_delta' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['mm_book_delta'], name='MM Option Book Delta', line=dict(dash='dot')), row=1, col=1)
    if 'current_n1' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['current_n1'], name='Underlying Hedge Qty (n1)'), row=2, col=1, secondary_y=False) 
    if 'spot_price' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['spot_price'], name='Spot Price'), row=2, col=1, secondary_y=True)
    if 'cumulative_pnl' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['cumulative_pnl'], name='Cumulative P&L'), row=3, col=1, secondary_y=False)
    if 'cumulative_trading_costs' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['cumulative_trading_costs'], name='Cumulative Costs', line=dict(dash='dot')), row=3, col=1, secondary_y=True)
    fig.update_layout(height=800, legend_title_text='Metrics'); st.plotly_chart(fig, use_container_width=True)

def plot_mm_delta_gamma_hedge(portfolio_state_df, hedge_actions_df, symbol): # Unchanged
    st.subheader(f"MM Delta-Gamma Hedging Simulation ({symbol} - Short Book)")
    if portfolio_state_df.empty: st.info("No data for MM Delta-Gamma hedge plot."); return
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{}],[{"secondary_y":True}]], subplot_titles=("Net Portfolio Greeks", f"Underlying Hedge & Spot", "Gamma Option Hedge", "Cumulative P&L and Costs"))
    if 'net_delta_final' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['net_delta_final'], name='Net Delta'), row=1, col=1, secondary_y=False)
    if 'net_gamma_final' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['net_gamma_final'], name='Net Gamma'), row=1, col=1, secondary_y=True)
    if 'current_n_underlying' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['current_n_underlying'], name='Underlying Qty'), row=2, col=1, secondary_y=False)
    if 'spot_price' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['spot_price'], name='Spot Price'), row=2, col=1, secondary_y=True)
    if 'current_n_gamma_opt' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['current_n_gamma_opt'], name='Gamma Opt Qty'), row=3, col=1)
    if 'cumulative_pnl' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['cumulative_pnl'], name='Cumulative P&L'), row=4, col=1, secondary_y=False)
    if 'cumulative_trading_costs' in portfolio_state_df: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['cumulative_trading_costs'], name='Cumulative Costs', line=dict(dash='dot')), row=4, col=1, secondary_y=True)
    fig.update_layout(height=1000, legend_title_text='Metrics'); st.plotly_chart(fig, use_container_width=True)
# --- display_mm_gamma_adjustment_analysis (Unchanged) ---
def display_mm_gamma_adjustment_analysis(dft_latest_snap, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    st.subheader("MM Indicative Delta-Gamma Hedge Adjustment (Selected Expiry)")
    if dft_latest_snap.empty or pd.isna(spot_price) or spot_price <= 0: st.info("Data insufficient for MM D-G adjustment analysis."); return
    df_book = dft_latest_snap.copy(); df_book['open_interest'] = pd.to_numeric(df_book['open_interest'], errors='coerce').fillna(0); df_book = df_book[df_book['open_interest'] > 0]
    if df_book.empty: st.info("No Open Interest in selected book for MM D-G adjustment."); return
    
    df_book['mm_delta_pos'] = -1 * df_book.apply(lambda r: compute_delta(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1) * df_book['open_interest']
    df_book['mm_gamma_pos'] = -1 * df_book.apply(lambda r: compute_gamma(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1) * df_book['open_interest']
    
    mm_delta_initial = df_book['mm_delta_pos'].sum(skipna=True); mm_gamma_initial = df_book['mm_gamma_pos'].sum(skipna=True)
    if pd.isna(mm_delta_initial) or pd.isna(mm_gamma_initial): st.warning("Could not calculate initial MM greeks for D-G adjustment."); return
    
    st.metric("MM Initial Net Delta", f"{mm_delta_initial:,.2f}"); st.metric("MM Initial Net Gamma", f"{mm_gamma_initial:,.4f}")
    
    calls_book = df_book[df_book['option_type'] == 'C'].copy()
    if calls_book.empty: st.warning("No calls available in the book to select as gamma hedger."); return
    
    calls_book['moneyness_dist'] = abs(calls_book['k'] - spot_price)
    hedger_cands = calls_book[calls_book['k'] >= spot_price].sort_values('moneyness_dist') 
    
    hedger_row_series = pd.Series(dtype='object') 
    if not hedger_cands.empty:
        hedger_row_series = hedger_cands.iloc[0]
    elif not calls_book.empty: 
        hedger_row_series = calls_book.sort_values('moneyness_dist').iloc[0]
    
    if hedger_row_series.empty: 
        st.warning("Could not select any call option as a gamma hedger."); return

    st.info(f"Selected Gamma Hedger: {hedger_row_series['instrument_name']} (K={hedger_row_series['k']})")
    
    D_h = compute_delta(hedger_row_series, spot_price, snapshot_time_utc, risk_free_rate)
    G_h = compute_gamma(hedger_row_series, spot_price, snapshot_time_utc, risk_free_rate)
    
    if pd.isna(D_h) or pd.isna(G_h) or abs(G_h) < 1e-9: st.error("Selected hedger greeks are invalid or gamma is too small."); return
    
    N_h = -mm_gamma_initial / G_h 
    delta_from_gh = N_h * D_h; mm_delta_post_gh = mm_delta_initial + delta_from_gh
    underlying_hedge = -mm_delta_post_gh
    
    cols_gh = st.columns(3)
    with cols_gh[0]: st.metric("Hedger Δ", f"{D_h:.4f}")
    with cols_gh[1]: st.metric("Hedger Γ", f"{G_h:.6f}")
    with cols_gh[2]: st.metric(f"Hedge Opt Qty ({'Buy' if N_h>0 else 'Sell'})", f"{abs(N_h):,.2f}")
    
    st.metric("Δ from Γ Hedge", f"{delta_from_gh:,.2f}")
    st.metric("MM Net Δ (Post Γ Hedge)", f"{mm_delta_post_gh:,.2f}")
    st.metric(f"Final Underlying Hedge ({'Buy' if underlying_hedge>0 else 'Sell'} Spot)", f"{abs(underlying_hedge):,.2f}")
    st.success(f"**Resulting Book Net Δ (Final):** {mm_delta_post_gh + underlying_hedge:,.4f}") 
    st.success(f"**Resulting Book Net Γ (Final):** {mm_gamma_initial + N_h * G_h:,.4f}") 
# --- Main Function ---
def main():
    st.set_page_config(layout="wide", page_title="Advanced Options Hedging & MM Dashboard")
    login()
    
    if 'selected_coin' not in st.session_state: st.session_state.selected_coin = "BTC"
    if 'snapshot_time' not in st.session_state: st.session_state.snapshot_time = dt.datetime.now(dt.timezone.utc)
    if 'risk_free_rate_input' not in st.session_state: st.session_state.risk_free_rate_input = 0.01
    if 'days_history_fetch' not in st.session_state: st.session_state.days_history_fetch = 7 # Default lookback
    
    st.title(f"{st.session_state.selected_coin} Options: Advanced Hedging & MM Perspective")
    if st.sidebar.button("Logout"): st.session_state.logged_in = False; st.rerun()

    st.sidebar.header("Configuration")
    coin_options = ["BTC", "ETH"]
    current_coin_idx = coin_options.index(st.session_state.selected_coin) if st.session_state.selected_coin in coin_options else 0
    selected_coin_widget = st.sidebar.selectbox("Cryptocurrency", coin_options, index=current_coin_idx, key="main_coin_select_adv")
    if selected_coin_widget != st.session_state.selected_coin:
        st.session_state.selected_coin = selected_coin_widget
        st.session_state.days_history_fetch = 7 # Reset lookback on coin change
        st.rerun()
    coin = st.session_state.selected_coin

    st.session_state.risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (Annualized)", value=st.session_state.risk_free_rate_input, min_value=0.0, max_value=0.2, step=0.001, format="%.3f", key="main_rf_rate_adv")
    risk_free_rate = st.session_state.risk_free_rate_input
    
    # Add configurable days history for options data
    days_history_options = [7, 14, 30, 60, 90]
    current_days_idx = days_history_options.index(st.session_state.days_history_fetch) if st.session_state.days_history_fetch in days_history_options else 0
    selected_days_history = st.sidebar.selectbox("Options Hist. Lookback (Days)", days_history_options, index=current_days_idx, key=f"days_hist_fetch_adv_{coin}")
    if selected_days_history != st.session_state.days_history_fetch:
        st.session_state.days_history_fetch = selected_days_history
        st.rerun() # Rerun to re-fetch data with new lookback
    
    st.session_state.snapshot_time = dt.datetime.now(dt.timezone.utc)

    all_instruments_list = fetch_instruments()
    if not all_instruments_list: st.error("Failed to fetch instruments."); st.stop()
    valid_expiries = get_valid_expiration_options(st.session_state.snapshot_time)
    if not valid_expiries: st.error(f"No valid expiries for {coin}."); st.stop()
    
    default_exp_idx = min(1, len(valid_expiries)-1) if len(valid_expiries)>1 else 0 
    selected_expiry_key = f"main_expiry_select_adv_{coin}_{st.session_state.days_history_fetch}" # Make key unique to avoid issues if other params change
    selected_expiry = st.sidebar.selectbox("Choose Expiry", valid_expiries, index=default_exp_idx, format_func=lambda d: d.strftime("%d %b %Y"), key=selected_expiry_key)
    e_str = selected_expiry.strftime("%d%b%y").upper() if selected_expiry else "N/A"
    
    all_calls_expiry = get_option_instruments(all_instruments_list, "C", e_str, coin)
    all_puts_expiry = get_option_instruments(all_instruments_list, "P", e_str, coin)
    all_instr_selected_expiry = sorted(all_calls_expiry + all_puts_expiry)

    st.sidebar.markdown("---"); st.sidebar.subheader("Pair Delta Hedge Sim")
    sel_call_pair = st.sidebar.selectbox("OTM Call (Pair Sim):", options=sorted(all_calls_expiry, key=lambda x: int(x.split('-')[2])), index=min(len(all_calls_expiry)-1, int(len(all_calls_expiry)*0.7)) if all_calls_expiry else 0, key=f"main_pair_call_adv_{e_str}_{coin}", disabled=not all_calls_expiry)
    sel_put_pair = st.sidebar.selectbox("OTM Put (Pair Sim):", options=sorted(all_puts_expiry, key=lambda x: int(x.split('-')[2])), index=max(0, int(len(all_puts_expiry)*0.3)-1) if all_puts_expiry else 0, key=f"main_pair_put_adv_{e_str}_{coin}", disabled=not all_puts_expiry)

    df_krak_5m = fetch_kraken_data(coin=coin, days=max(30, st.session_state.days_history_fetch + 5)) # Fetch a bit more spot history
    spot_price = df_krak_5m["close"].iloc[-1] if not df_krak_5m.empty else np.nan
    st.header(f"Analysis: {coin} | Expiry: {e_str} | Spot: ${spot_price:,.2f}" if pd.notna(spot_price) else f"Analysis: {coin} | Expiry: {e_str} | Spot: N/A")
    st.markdown(f"*Snapshot: {st.session_state.snapshot_time.strftime('%Y-%m-%d %H:%M:%S UTC')} | RF Rate: {risk_free_rate:.3%} | Opt. Hist. Lookback: {st.session_state.days_history_fetch} days*")

    if not all_instr_selected_expiry: st.error(f"No options for {e_str}."); st.stop()
    
    # Fetch data for selected expiry using configured days_history_fetch
    dft_raw = fetch_data(tuple(all_instr_selected_expiry), days_history=st.session_state.days_history_fetch) 
    
    ticker_data = {instr: fetch_ticker(instr) for instr in all_instr_selected_expiry}
    valid_tickers = {k for k,v in ticker_data.items() if v and v.get('iv',0) > 1e-4 and pd.notna(v.get('open_interest'))}
    
    dft = dft_raw[dft_raw['instrument_name'].isin(valid_tickers)].copy() if not dft_raw.empty else pd.DataFrame()
    if not dft.empty: 
        dft['open_interest'] = dft['instrument_name'].map(lambda x: ticker_data.get(x,{}).get('open_interest',0.0))
        dft['iv_close'] = pd.to_numeric(dft['iv_close'], errors='coerce') # Redundant if done in fetch_data, but safe
    else: st.warning(f"No valid historical option data after ticker filtering for the selected expiry with {st.session_state.days_history_fetch}-day lookback.")

    dft_with_hist_greeks = pd.DataFrame()
    if not dft.empty and not df_krak_5m.empty:
        merged_hist = pd.merge_asof(dft.sort_values('date_time'), df_krak_5m[['date_time','close']].rename(columns={'close':'spot_hist'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('10min')).dropna(subset=['spot_hist'])
        if not merged_hist.empty:
            with st.spinner("Calculating Greeks on historical data for simulations..."):
                merged_hist['delta'] = merged_hist.apply(lambda r: compute_delta(r, r['spot_hist'], r['date_time'], risk_free_rate), axis=1)
                merged_hist['gamma'] = merged_hist.apply(lambda r: compute_gamma(r, r['spot_hist'], r['date_time'], risk_free_rate), axis=1)
                dft_with_hist_greeks = merged_hist.dropna(subset=['delta', 'gamma', 'iv_close']) 
    
    if dft_with_hist_greeks.empty:
        st.warning(f"Could not generate `dft_with_hist_greeks`. This usually means sparse/missing historical IV data for the selected options within the {st.session_state.days_history_fetch}-day lookback. Simulations might be impacted or skipped.")
        dft_with_hist_greeks = dft.copy() # Fallback, greeks might be missing or based on current spot
        if not dft_with_hist_greeks.empty and pd.notna(spot_price): 
             if 'delta' not in dft_with_hist_greeks.columns:
                 dft_with_hist_greeks["delta"] = dft_with_hist_greeks.apply(lambda row: compute_delta(row, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
             if 'gamma' not in dft_with_hist_greeks.columns:
                 dft_with_hist_greeks["gamma"] = dft_with_hist_greeks.apply(lambda row: compute_gamma(row, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)

    dft_latest = pd.DataFrame()
    if not dft.empty and pd.notna(spot_price):
        if 'ts' in dft.columns:
            latest_indices = dft.groupby('instrument_name')['ts'].idxmax()
            dft_latest_raw = dft.loc[latest_indices].copy()
        
            if not dft_latest_raw.empty:
                with st.spinner("Calculating Greeks for latest snapshot..."):
                    dft_latest_raw['delta'] = dft_latest_raw.apply(lambda r: compute_delta(r, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
                    dft_latest_raw['gamma'] = dft_latest_raw.apply(lambda r: compute_gamma(r, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
                    dft_latest_raw['vega'] = dft_latest_raw.apply(lambda r: compute_vega(r, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
                    dft_latest_raw['charm'] = dft_latest_raw.apply(lambda r: compute_charm(r, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
                    dft_latest_raw['vanna'] = dft_latest_raw.apply(lambda r: compute_vanna(r, spot_price, st.session_state.snapshot_time, risk_free_rate), axis=1)
                    dft_latest_raw['gex'] = dft_latest_raw.apply(lambda r: compute_gex(r, spot_price, r.get('open_interest', np.nan)), axis=1) 
                    dft_latest = dft_latest_raw.dropna(subset=['delta','gamma','vega', 'gex', 'iv_close']) 
    
    def safe_plot_exec(plot_func, *args, **kwargs):
        try: plot_func(*args, **kwargs)
        except Exception as e: st.error(f"Plot error in {plot_func.__name__}: {e}"); logging.error(f"Plot error in {plot_func.__name__}", exc_info=True)
    
    if not dft_latest.empty: 
        ticker_list_latest_snap = build_ticker_list(dft_latest, ticker_data)
        if ticker_list_latest_snap:
            safe_plot_exec(plot_open_interest_delta, ticker_list_latest_snap, spot_price)
            safe_plot_exec(plot_delta_balance, ticker_list_latest_snap, spot_price)
    
    st.markdown("---"); st.header("1. Delta Hedging Simulations")
    if st.sidebar.checkbox("Show Delta Hedging Simulations", True, key="adv_show_dh_sims"):
        if not dft_with_hist_greeks.empty and not df_krak_5m.empty and pd.notna(spot_price):
            tc_bps_val = st.sidebar.number_input("Transaction Cost (bps)", 0, 10, TRANSACTION_COST_BPS, 1, key="adv_tc_bps_val")
            use_dyn_thresh_std_val = st.sidebar.checkbox("Dynamic Threshold (Std.)", False, key="adv_dh_dyn_thr_val")
            sim_days_lookback = st.sidebar.slider("Simulation History (Days)", 1, min(30, st.session_state.days_history_fetch), 7, key="sim_days_lookback_slider")


            hedge_instance_std = HedgeThalex(dft_with_hist_greeks, df_krak_5m, coin, use_dynamic_threshold=use_dyn_thresh_std_val, transaction_cost_bps=tc_bps_val)
            delta_df_std, hedge_actions_std = hedge_instance_std.run_loop(days=sim_days_lookback) 
            if not delta_df_std.empty : 
                safe_plot_exec(plot_delta_hedging_thalex, delta_df_std, hedge_actions_std, hedge_instance_std.base_threshold, use_dyn_thresh_std_val, coin, spot_price, df_krak_5m)
            else:
                st.info(f"Traditional Delta Hedging (HedgeThalex): Simulation produced no data for the last {sim_days_lookback} days. This might be due to insufficient overlapping historical option and spot data. Try increasing the 'Options Hist. Lookback (Days)' in sidebar and/or adjusting 'Simulation History (Days)'.")

            matrix_instance_std = MatrixHedgeThalex(dft_with_hist_greeks, df_krak_5m, coin, transaction_cost_bps=tc_bps_val)
            state_df_mat_std, actions_df_mat_std = matrix_instance_std.run_loop(days=sim_days_lookback)
            if not state_df_mat_std.empty: 
                safe_plot_exec(plot_matrix_hedge_thalex, state_df_mat_std, actions_df_mat_std, coin)
            else:
                st.info("Matrix Delta Hedging: Simulation produced no data.")


            if sel_call_pair and sel_put_pair:
                safe_plot_exec(plot_net_delta_otm_pair, 
                               dft=dft_with_hist_greeks, # dft_with_hist_greeks is already filtered by expiry and uses days_history_fetch
                               df_spot_hist=df_krak_5m, 
                               exchange_instance=exchange1, 
                               selected_call_instr=sel_call_pair, 
                               selected_put_instr=sel_put_pair,
                               all_available_calls_for_expiry=all_calls_expiry,
                               all_available_puts_for_expiry=all_puts_expiry)
            else:
                st.info("Select both Call and Put for OTM Pair Delta Simulation.")
        else: st.warning(f"Data insufficient for Delta Hedging Simulations (dft_with_hist_greeks is empty, or df_krak_5m/spot_price missing). Please check 'Options Hist. Lookback (Days)' or select a different expiry if problems persist for long-dated options.")

    st.markdown("---"); st.header(f"2. ITM Gamma Exposure Analysis (Expiry: {e_str})")
    if not dft_with_hist_greeks.empty and not df_krak_5m.empty and pd.notna(spot_price) and selected_expiry:
        plot_data_for_gex_adv = compute_and_plot_itm_gex_ratio(dft=dft_with_hist_greeks, df_krak_5m=df_krak_5m, spot_price_latest=spot_price, selected_expiry_obj=selected_expiry)
    else: st.warning("Data insufficient for ITM GEX Analysis.")

    st.markdown("---"); st.header("3. Market Maker Perspective")
    if not dft_latest.empty:
        st.subheader("Net Greek Exposures (Latest Snapshot - MM Short Book)")
        cols_greeks_adv = st.columns(5)
        net_d_mm = -(dft_latest['delta'] * dft_latest.get('open_interest',0)).sum(); cols_greeks_adv[0].metric("Net Delta", f"{net_d_mm:,.2f}" if pd.notna(net_d_mm) else "N/A")
        net_g_mm = -(dft_latest['gamma'] * dft_latest.get('open_interest',0)).sum(); cols_greeks_adv[1].metric("Net Gamma", f"{net_g_mm:,.4f}" if pd.notna(net_g_mm) else "N/A")
        net_v_mm = -(dft_latest['vega'] * dft_latest.get('open_interest',0)).sum(); cols_greeks_adv[2].metric("Net Vega", f"{net_v_mm:,.0f}" if pd.notna(net_v_mm) else "N/A")
        net_va_mm = -(dft_latest['vanna'] * dft_latest.get('open_interest',0)).sum(); cols_greeks_adv[3].metric("Net Vanna", f"{net_va_mm:,.0f}" if pd.notna(net_va_mm) else "N/A")
        net_ch_mm = -(dft_latest['charm'] * dft_latest.get('open_interest',0)).sum(); cols_greeks_adv[4].metric("Net Charm", f"{net_ch_mm:,.2f}" if pd.notna(net_ch_mm) else "N/A")

        safe_plot_exec(display_mm_gamma_adjustment_analysis, dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)
    
    show_mm_dg_sim_main = st.sidebar.checkbox("Show MM D-G Hedge Sim Plot", value=True, key="show_mm_dg_sim_main_plot_v5") 

    mm_dg_portfolio_state_df_sim = pd.DataFrame()
    mm_dg_hedge_actions_df_sim = pd.DataFrame()

    if show_mm_dg_sim_main:
        st.markdown("---") 
        st.header("MM Delta-Gamma Hedging Simulation (Historical)") 
        sim_days_lookback_dg = st.sidebar.slider("MM D-G Sim History (Days)", 1, min(30, st.session_state.days_history_fetch), 7, key="sim_days_dg_lookback_slider")

        if not dft_with_hist_greeks.empty and not df_krak_5m.empty and selected_expiry and not dft_latest.empty:
            gamma_hedger_candidate_df = dft_latest[
                (dft_latest['option_type'] == 'C') &
                (dft_latest['k'] >= spot_price) 
            ].sort_values('k') 

            if gamma_hedger_candidate_df.empty: 
                all_calls_latest = dft_latest[dft_latest['option_type'] == 'C']
                if not all_calls_latest.empty:
                    gamma_hedger_candidate_df = all_calls_latest.loc[[
                        abs(all_calls_latest['k'] - spot_price).idxmin()
                    ]]
                    logging.info("MM D-G Sim: No ATM/OTM calls found. Using closest ITM call as fallback for gamma hedger.")
                else:
                    logging.warning("MM D-G Sim: No call options found in dft_latest for the selected expiry to pick hedger.")
            
            if not gamma_hedger_candidate_df.empty:
                gamma_hedger_row_sim = gamma_hedger_candidate_df.iloc[0]
                
                gamma_hedger_details_sim = {
                    'name': gamma_hedger_row_sim['instrument_name'],
                    'k': gamma_hedger_row_sim['k'],
                    'option_type': gamma_hedger_row_sim['option_type'],
                    'expiry_datetime_col': gamma_hedger_row_sim['expiry_datetime_col'] 
                }
                logging.info(f"MM D-G Sim: Selected gamma hedging instrument: {gamma_hedger_details_sim['name']}")

                try:
                    mm_dg_sim_instance = MatrixDeltaGammaHedgeSimple(
                        df_portfolio_options=dft_with_hist_greeks, 
                        spot_df=df_krak_5m,             
                        symbol=coin,
                        risk_free_rate=risk_free_rate,
                        gamma_hedge_instrument_details=gamma_hedger_details_sim
                    )
                    logging.info("MM D-G Sim: MatrixDeltaGammaHedgeSimple instance created.")

                    with st.spinner("Running MM Delta-Gamma Hedge Simulation..."):
                        mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim = mm_dg_sim_instance.run_loop(days=sim_days_lookback_dg)
                    
                    logging.info(f"MM D-G Sim: run_loop completed. Portfolio states: {len(mm_dg_portfolio_state_df_sim)}, Hedge actions: {len(mm_dg_hedge_actions_df_sim)}")

                    if not mm_dg_portfolio_state_df_sim.empty:
                        safe_plot_exec(plot_mm_delta_gamma_hedge, mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim, coin)
                    else:
                        st.info("MM D-G Sim: Simulation ran but produced no portfolio state data to plot.")
                        logging.info("MM D-G Sim: mm_dg_portfolio_state_df_sim is empty after run_loop.")

                except ValueError as ve_init_sim: 
                    st.error(f"MM D-G Sim Initialization Error: {ve_init_sim}")
                    logging.error(f"MM D-G Sim: ValueError during MatrixDeltaGammaHedgeSimple initialization: {ve_init_sim}", exc_info=True)
                except Exception as e_mm_sim: 
                    st.error(f"MM D-G Sim Runtime Error: {e_mm_sim}")
                    logging.error(f"MM D-G Sim: Exception during run or init: {e_mm_sim}", exc_info=True)
            
            else: 
                st.warning("MM D-G Sim: Could not find any suitable Call option in the latest snapshot to use as a gamma hedging instrument for the simulation.")
                logging.warning("MM D-G Sim: gamma_hedger_candidate_df remained empty. Cannot run simulation.")
        
        else: 
            missing_data_reasons = []
            if dft_with_hist_greeks.empty: missing_data_reasons.append("'dft_with_hist_greeks' is empty")
            if df_krak_5m.empty: missing_data_reasons.append("'df_krak_5m' is empty")
            if not selected_expiry: missing_data_reasons.append("'selected_expiry' is not set")
            if dft_latest.empty: missing_data_reasons.append("'dft_latest' is empty")
            
            st.warning(f"MM D-G Sim: Cannot run simulation due to missing data: {', '.join(missing_data_reasons)}. Consider increasing 'Options Hist. Lookback (Days)'.")
            logging.warning(f"MM D-G Sim: Pre-conditions not met. Reasons: {', '.join(missing_data_reasons)}")

    
    st.markdown("---"); st.header("Raw Data Tables")
    with st.expander("Latest Snapshot Data (dft_latest)"):
        if not dft_latest.empty: st.dataframe(dft_latest[[c for c in ['instrument_name', 'k', 'option_type', 'mark_price_close', 'iv_close', 'delta', 'gamma', 'vega', 'vanna', 'charm', 'gex', 'open_interest'] if c in dft_latest.columns]].round(4))
        else: st.write("No latest snapshot data available.")
    with st.expander("Historical Data with Greeks for Sims (dft_with_hist_greeks - Head)"):
        if not dft_with_hist_greeks.empty: st.dataframe(dft_with_hist_greeks.head().round(4))
        else: st.write("No historical data with Greeks available.")

    gc.collect()
    logging.info(f"ADVANCED Dashboard rendering complete for {coin} {e_str}.")

if __name__ == "__main__":
    main()
