import streamlit as st
import datetime as dt
import scipy.stats as si
import scipy.interpolate # For the delta interpolation helper
# from scipy.stats import linregress # Not needed for this focused version
# import statsmodels.tsa.stattools as smt # Not needed for this focused version
# import scipy # Already imported above
import pandas as pd
import requests
import numpy as np
import ccxt
# from toolz.curried import pipe, valmap, get_in, curry, valfilter # Not used in focused version
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
import logging
import time
from plotly.subplots import make_subplots
import math
import gc
from scipy.stats import linregress # Needed for Hurst

# RF Regressor and related ML imports are removed as IV modeling is out of scope

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

# --- Utility Functions ---

## Login Functions
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user: users = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass: pwds = [line.strip() for line in f_pass if line.strip()]
        if len(users) != len(pwds): st.error("Username/password mismatch."); return {}
        return dict(zip(users, pwds))
    except FileNotFoundError: st.error("Credential files not found."); return {}
    except Exception as e: st.error(f"Credential load error: {e}"); return {}

def login():
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Log In")
        creds = load_credentials()
        if not creds: st.stop()
        username = st.text_input("Username"); password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True; st.rerun()
            else: st.error("Invalid credentials")
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
    except Exception: return []

@st.cache_data(ttl=30)
def fetch_ticker(instr_name):
    try:
        r = requests.get(URL_TICKER, params={"instrument_name": instr_name}, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        return r.json().get("result", {})
    except Exception: return None

def params_historical(instrument_name, days=7):
    now = dt.datetime.now(dt.timezone.utc); start_dt = now - dt.timedelta(days=days)
    return {"from": int(start_dt.timestamp()), "to": int(now.timestamp()), "resolution": "5m", "instrument_name": instrument_name}

@st.cache_data(ttl=60)
def fetch_data(instruments_tuple):
    instr = list(instruments_tuple)
    if not instr: return pd.DataFrame()
    dfs = []
    errors = 0
    for name in instr:
        resp = None
        try:
            params_req = params_historical(name)
            resp = requests.get(URL_MARK_PRICE, params=params_req, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            marks = safe_get_in(["result", "mark"], data, default=[])
            if marks:
                df_temp = pd.DataFrame(marks, columns=COLUMNS)
                df_temp["instrument_name"] = name
                if not df_temp.empty and "ts" in df_temp.columns and df_temp["ts"].notna().any():
                    dfs.append(df_temp)
        except requests.exceptions.RequestException as e:
            logging.error(f"Network/HTTP error fetching data for {name}: {e}")
            errors += 1
        except Exception as e:
            logging.error(f"Error processing data for {name}: {e}", exc_info=True)
            errors += 1
        time.sleep(0.1)

    if errors > 0:
        st.warning(f"Encountered errors fetching data for {errors}/{len(instr)} instruments. Check logs.")

    if not dfs: return pd.DataFrame()
    try:
        dfc = pd.concat(dfs).reset_index(drop=True)
        dfc['date_time'] = pd.to_datetime(dfc['ts'], unit='s', errors='coerce').dt.tz_localize('UTC')
        dfc = dfc.dropna(subset=['date_time'])

        def safe_get_strike(s):
            try:
                return int(s.split('-')[2])
            except:
                return np.nan

        def safe_get_type(s):
            try:
                return s.split('-')[-1]
            except:
                return None

        dfc['k'] = dfc['instrument_name'].apply(safe_get_strike)
        dfc['option_type'] = dfc['instrument_name'].apply(safe_get_type)

        def get_expiry_datetime_from_name(instr_name_str):
            try:
                if not isinstance(instr_name_str, str): return pd.NaT
                parts = instr_name_str.split('-')
                if len(parts) >= 2:
                    return dt.datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8, minute=0, second=0)
            except Exception:
                pass
            return pd.NaT

        dfc['expiry_datetime_col'] = dfc['instrument_name'].apply(get_expiry_datetime_from_name)
        essential_cols = ['date_time', 'k', 'option_type', 'mark_price_close', 'iv_close', 'expiry_datetime_col']
        dfc = dfc.dropna(subset=essential_cols)

        if dfc.empty:
            logging.warning("DataFrame empty after essential column NaN drop in fetch_data.")
            return pd.DataFrame()
        return dfc.sort_values("date_time")
    except Exception as e:
        st.error(f"Error during final processing in fetch_data: {e}")
        logging.error(f"Final processing error in fetch_data: {e}", exc_info=True)
        return pd.DataFrame()


exchange1 = None
try:
    exchange1 = ccxt.bitget({'enableRateLimit': True})
except Exception as e:
    st.error(f"⚠️ Failed to connect to Bitget: {e}")

def fetch_funding_rates(exchange_instance, symbol='BTC/USDT', start_time=None, end_time=None):
    if exchange_instance is None: return pd.DataFrame(columns=['date_time', 'raw_funding_rate', 'funding_rate'])
    try:
        markets = exchange_instance.load_markets()
        if symbol not in markets: return pd.DataFrame(columns=['date_time', 'raw_funding_rate', 'funding_rate'])
        since = int(start_time.timestamp() * 1000) if start_time else None
        funding_rates_hist = exchange_instance.fetch_funding_rate_history(symbol=symbol, since=since)
        if not funding_rates_hist: return pd.DataFrame(columns=['date_time', 'raw_funding_rate', 'funding_rate'])
        funding_data = [{'date_time': pd.to_datetime(entry['timestamp'], unit='ms', utc=True), 'raw_funding_rate': entry['fundingRate'], 'funding_rate': entry['fundingRate'] * (365 * 3)} for entry in funding_rates_hist if not (end_time and pd.to_datetime(entry['timestamp'], unit='ms', utc=True) > end_time)]
        return pd.DataFrame(funding_data)
    except Exception: return pd.DataFrame(columns=['date_time', 'raw_funding_rate', 'funding_rate'])

def fetch_kraken_data(coin="BTC", days=7):
    try:
        k = ccxt.kraken(); now_dt = dt.datetime.now(dt.timezone.utc); start_dt = now_dt - dt.timedelta(days=days)
        since = int(start_dt.timestamp() * 1000); symbol = f"{coin}/USD"
        ohlcv = k.fetch_ohlcv(symbol, timeframe="5m", since=since)
        if not ohlcv: return pd.DataFrame()
        dfr = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dfr["date_time"] = pd.to_datetime(dfr["timestamp"], unit="ms", errors='coerce').dt.tz_localize("UTC")
        dfr = dfr.dropna(subset=['date_time']).sort_values("date_time")
        return dfr[dfr["date_time"] >= start_dt].reset_index(drop=True)
    except Exception: return pd.DataFrame()

def fetch_kraken_data_daily(days=365, coin="BTC"): # RE-ADDED THIS FUNCTION
    """Fetch daily OHLCV data from Kraken. Returns UTC-aware DataFrame."""
    logging.info(f"Fetching {days} days of daily Kraken data for {coin}/USD.")
    try:
        k = ccxt.kraken()
        now_dt = dt.datetime.now(dt.timezone.utc) # Ensure UTC
        start_dt = now_dt - dt.timedelta(days=days)
        since = int(start_dt.timestamp() * 1000)
        symbol = f"{coin}/USD"
        ohlcv = k.fetch_ohlcv(symbol, timeframe="1d", since=since)
        if not ohlcv:
            st.warning(f"No daily OHLCV data returned from Kraken for {symbol}.")
            return pd.DataFrame()

        dfr = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dfr["date_time"] = pd.to_datetime(dfr["timestamp"], unit="ms", errors='coerce').dt.tz_localize("UTC")
        dfr = dfr.dropna(subset=['date_time'])
        dfr.sort_values("date_time", inplace=True)
        dfr.reset_index(drop=True, inplace=True)
        logging.info(f"Kraken daily data fetched. Shape: {dfr.shape}")
        return dfr

    except ccxt.NetworkError as e:
        st.error(f"Kraken Network Error fetching daily data: {e}")
        logging.error(f"Kraken Network Error fetching daily data: {e}")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        st.error(f"Kraken Exchange Error fetching daily data: {e}")
        logging.error(f"Kraken Exchange Error fetching daily data: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error fetching Kraken daily data: {e}")
        logging.error(f"Unexpected error fetching Kraken daily data: {e}", exc_info=True)
        return pd.DataFrame()

def get_valid_expiration_options(current_date_utc):
    instruments = fetch_instruments()
    if not instruments: return []
    exp_dates = set()
    for instr in instruments:
        parts = instr.get("instrument_name", "").split("-")
        if len(parts) < 3 or not parts[-1] in ['C', 'P']: continue
        try:
            expiry_date = dt.datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
            if expiry_date > current_date_utc: exp_dates.add(expiry_date)
        except ValueError: continue
    return sorted(list(exp_dates))

def get_option_instruments(instruments, option_type, expiry_str, coin):
    return sorted([i["instrument_name"] for i in instruments if (i.get("instrument_name", "").startswith(f"{coin}-{expiry_str}") and i.get("instrument_name", "").endswith(f"-{option_type}"))])

def merge_spot_to_options(dft_options, df_spot, expiry_dt): # Keep for historical greek calcs
    if dft_options.empty or df_spot.empty: return pd.DataFrame()
    dft_local = dft_options.copy(); spot_local = df_spot[['date_time', 'close']].copy()
    for df in [dft_local, spot_local]:
        if not pd.api.types.is_datetime64_any_dtype(df['date_time']): return pd.DataFrame()
        if df['date_time'].dt.tz is None: df['date_time'] = df['date_time'].dt.tz_localize('UTC')
        elif df['date_time'].dt.tz != dt.timezone.utc: df['date_time'] = df['date_time'].dt.tz_convert('UTC')
    dft_local = dft_local.sort_values('date_time'); spot_local = spot_local.sort_values('date_time')
    df_merged = pd.merge_asof(dft_local, spot_local.rename(columns={'close': 'spot_price'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
    SECONDS_PER_YEAR = 365.0 * 24 * 3600
    df_merged['T_years'] = (expiry_dt - df_merged['date_time']).dt.total_seconds() / SECONDS_PER_YEAR
    df_merged = df_merged[df_merged['T_years'] > 1e-9]
    essential_cols = ['date_time', 'spot_price', 'k', 'mark_price_close', 'option_type', 'T_years']
    df_merged = df_merged.dropna(subset=essential_cols)
    for col in ['spot_price', 'k', 'mark_price_close', 'T_years']: df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    df_merged = df_merged.dropna(subset=['spot_price', 'k', 'mark_price_close', 'T_years'])
    return df_merged

# --- Realized Volatility Functions (for Key Metrics) ---
def compute_realized_volatility_5min(df, annualize_days=365):
    if df.empty or 'close' not in df.columns or len(df) < 2: return 0.0
    df = df.copy(); df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df_valid = df.dropna(subset=['log_ret'])
    if df_valid.empty: return 0.0
    total_variance = df_valid['log_ret'].pow(2).sum()
    if total_variance <= 0: return 0.0
    N = len(df_valid); M = annualize_days * 24 * 12
    return np.sqrt(total_variance) * (np.sqrt(M / N) if N > 0 else 0)

def calculate_btc_annualized_volatility_daily(df_daily):
    """Calculate annualized volatility for BTC using daily data over the last 30 days."""
    if df_daily.empty or 'close' not in df_daily.columns or len(df_daily) < 2:
        return np.nan
    df_daily = df_daily.dropna(subset=["close"]).copy()
    if len(df_daily) < 2:
        return np.nan
    df_daily["log_return"] = np.log(df_daily["close"] / df_daily["close"].shift(1))
    last_n_days = 30
    if len(df_daily["log_return"].dropna()) < last_n_days:
        last_n_days = len(df_daily["log_return"].dropna())
    if last_n_days < 2:
        return np.nan
    last_period_returns = df_daily["log_return"].dropna().tail(last_n_days)
    return last_period_returns.std(ddof=1) * np.sqrt(365)


def compute_delta(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close'); option_type = row.get('option_type')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(option_type) or option_type not in ['C', 'P'] or pd.isna(sigma) or pd.isna(r): return np.nan
        expiry_date_from_col = row.get('expiry_datetime_col')
        if pd.notna(expiry_date_from_col) and isinstance(expiry_date_from_col, dt.datetime):
            expiry_date = expiry_date_from_col
            if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            elif expiry_date.tzinfo != dt.timezone.utc: expiry_date = expiry_date.tz_convert(dt.timezone.utc)
        else:
            expiry_str = instr_name.split("-")[1]; expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        elif snapshot_time_utc.tzinfo != dt.timezone.utc: snapshot_time_utc = snapshot_time_utc.tz_convert(dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)
        if sigma < 1e-7 or T < 1e-9: return 1.0 if option_type == 'C' and S > k else -1.0 if option_type == 'P' and S < k else 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if abs(sigma_sqrt_T) < 1e-12: return np.nan
        log_S_K = np.log(S / k)
        if not np.isfinite(log_S_K): return np.nan
        d1 = (log_S_K + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        if not np.isfinite(d1): return np.nan
        delta_val = si.norm.cdf(d1) if option_type == 'C' else si.norm.cdf(d1) - 1.0
        return delta_val if np.isfinite(delta_val) else np.nan
    except Exception: return np.nan

def compute_gamma(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(sigma) or pd.isna(r) or sigma < 1e-7: return np.nan if not (sigma < 1e-7) else 0.0
        expiry_date_from_col = row.get('expiry_datetime_col')
        if pd.notna(expiry_date_from_col) and isinstance(expiry_date_from_col, dt.datetime):
            expiry_date = expiry_date_from_col
            if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            elif expiry_date.tzinfo != dt.timezone.utc: expiry_date = expiry_date.tz_convert(dt.timezone.utc)
        else:
            expiry_str = instr_name.split("-")[1]; expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        elif snapshot_time_utc.tzinfo != dt.timezone.utc: snapshot_time_utc = snapshot_time_utc.tz_convert(dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)
        if T < 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if abs(sigma_sqrt_T) < 1e-12: return np.nan
        log_S_K = np.log(S / k)
        if not np.isfinite(log_S_K): return np.nan
        d1 = (log_S_K + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
        if not np.isfinite(d1): return np.nan
        pdf_d1 = norm.pdf(d1)
        if not np.isfinite(pdf_d1) or abs(S * sigma_sqrt_T) < 1e-12 : return np.nan
        gamma_val = pdf_d1 / (S * sigma_sqrt_T)
        return gamma_val if np.isfinite(gamma_val) else np.nan
    except Exception: return np.nan

def compute_vega(row, S, snapshot_time_utc): # r is not used in standard Vega if not pricing
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(sigma) or sigma <= 1e-7: return np.nan if not (sigma <= 1e-7) else 0.0
        expiry_date_from_col = row.get('expiry_datetime_col')
        if pd.notna(expiry_date_from_col) and isinstance(expiry_date_from_col, dt.datetime):
            expiry_date = expiry_date_from_col
            if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            elif expiry_date.tzinfo != dt.timezone.utc: expiry_date = expiry_date.tz_convert(dt.timezone.utc)
        else:
            expiry_str = instr_name.split("-")[1]; expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        elif snapshot_time_utc.tzinfo != dt.timezone.utc: snapshot_time_utc = snapshot_time_utc.tz_convert(dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)
        if T < 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if abs(sigma_sqrt_T) < 1e-12: return np.nan # Avoid division by zero
        # Using r=0 for d1 in Vega as per convention (Vega is sensitivity to vol, not directly to r)
        d1 = (np.log(S / k) + (0.5 * sigma**2) * T) / sigma_sqrt_T
        if not np.isfinite(d1): return np.nan
        vega = S * norm.pdf(d1) * sqrt_T * 0.01 # Vega for 1% change in vol
        return vega if np.isfinite(vega) else np.nan
    except Exception: return np.nan

def compute_charm(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close') # option_type = row.get('option_type')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(sigma) or pd.isna(r) or sigma < 1e-7: return np.nan if not (sigma < 1e-7) else 0.0
        expiry_date_from_col = row.get('expiry_datetime_col')
        if pd.notna(expiry_date_from_col) and isinstance(expiry_date_from_col, dt.datetime):
            expiry_date = expiry_date_from_col
            if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            elif expiry_date.tzinfo != dt.timezone.utc: expiry_date = expiry_date.tz_convert(dt.timezone.utc)
        else:
            expiry_str = instr_name.split("-")[1]; expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        elif snapshot_time_utc.tzinfo != dt.timezone.utc: snapshot_time_utc = snapshot_time_utc.tz_convert(dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)
        if T < 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if abs(sigma_sqrt_T) < 1e-12: return np.nan
        log_S_K = np.log(S / k)
        if not np.isfinite(log_S_K): return np.nan
        b = r # Assuming cost of carry b = risk-free rate r for charm
        d1 = (log_S_K + (b + 0.5 * sigma**2) * T) / sigma_sqrt_T; d2 = d1 - sigma_sqrt_T
        if not np.isfinite(d1) or not np.isfinite(d2): return np.nan
        pdf_d1 = norm.pdf(d1)
        if not np.isfinite(pdf_d1) or T <= 1e-9: return np.nan if not (T <= 1e-9) else 0.0 # Avoid div by zero for T
        charm_annual = -pdf_d1 * d2 / (2 * T) # Standard Charm formula (dDelta/dTheta) with b=r
        charm_daily = charm_annual / 365.0
        return charm_daily if np.isfinite(charm_daily) else np.nan
    except Exception: return np.nan

def compute_vanna(row, S, snapshot_time_utc, r=0.0):
    instr_name = row.get('instrument_name', 'N/A')
    try:
        k = row.get('k'); sigma = row.get('iv_close')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(sigma) or pd.isna(r) or sigma < 1e-7: return np.nan if not (sigma < 1e-7) else 0.0
        expiry_date_from_col = row.get('expiry_datetime_col')
        if pd.notna(expiry_date_from_col) and isinstance(expiry_date_from_col, dt.datetime):
            expiry_date = expiry_date_from_col
            if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=dt.timezone.utc)
            elif expiry_date.tzinfo != dt.timezone.utc: expiry_date = expiry_date.tz_convert(dt.timezone.utc)
        else:
            expiry_str = instr_name.split("-")[1]; expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=dt.timezone.utc, hour=8)
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
        elif snapshot_time_utc.tzinfo != dt.timezone.utc: snapshot_time_utc = snapshot_time_utc.tz_convert(dt.timezone.utc)
        T = (expiry_date - snapshot_time_utc).total_seconds() / (365.0 * 24.0 * 3600.0)
        if T < 1e-9: return 0.0
        sqrt_T = math.sqrt(T); sigma_sqrt_T = sigma * sqrt_T
        if abs(sigma_sqrt_T) < 1e-12 or abs(sigma) < 1e-7: return np.nan # Added check for sigma itself
        log_S_K = np.log(S / k)
        if not np.isfinite(log_S_K): return np.nan
        d1 = (log_S_K + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T; d2 = d1 - sigma_sqrt_T
        if not np.isfinite(d1) or not np.isfinite(d2) : return np.nan
        pdf_d1 = norm.pdf(d1)
        if not np.isfinite(pdf_d1) : return np.nan
        vanna = -math.exp(-r * T) * pdf_d1 * d2 / sigma # Standard Vanna formula
        return vanna if np.isfinite(vanna) else np.nan
    except Exception: return np.nan

def calculate_net_vega(df_latest_snap):
    if df_latest_snap.empty or 'vega' not in df_latest_snap.columns or 'open_interest' not in df_latest_snap.columns: return np.nan
    try: vega_oi = (pd.to_numeric(df_latest_snap['vega'], errors='coerce') * pd.to_numeric(df_latest_snap['open_interest'], errors='coerce')); return vega_oi.sum(skipna=True)
    except Exception: return np.nan

def calculate_net_vanna(df_latest_snap):
    if df_latest_snap.empty or 'vanna' not in df_latest_snap.columns or 'open_interest' not in df_latest_snap.columns: return np.nan
    try: vanna_oi = (pd.to_numeric(df_latest_snap['vanna'], errors='coerce') * pd.to_numeric(df_latest_snap['open_interest'], errors='coerce')); return vanna_oi.sum(skipna=True)
    except Exception: return np.nan

def calculate_net_charm(df_latest_snap):
    if df_latest_snap.empty or 'charm' not in df_latest_snap.columns or 'open_interest' not in df_latest_snap.columns: return np.nan
    try:
        charm_numeric = pd.to_numeric(df_latest_snap['charm'], errors='coerce'); oi_numeric = pd.to_numeric(df_latest_snap['open_interest'], errors='coerce')
        charm_oi = charm_numeric * oi_numeric; net_charm = charm_oi.sum(skipna=True)
        return net_charm if np.isfinite(net_charm) else np.nan
    except Exception: return np.nan

def compute_gex(row, S, oi):
    try:
        gamma_val = row.get('gamma'); oi_val = float(oi) if pd.notna(oi) else np.nan
        if pd.isna(gamma_val) or pd.isna(oi_val) or pd.isna(S) or S <= 0 or oi_val < 0: return np.nan
        gex = gamma_val * oi_val * (S ** 2) * 0.01
        return gex if np.isfinite(gex) else np.nan
    except Exception: return np.nan

def build_ticker_list(dft_latest, ticker_data): # For MM views if needed
    if dft_latest.empty: return []
    req_cols = ['instrument_name', 'k', 'option_type', 'delta', 'gamma', 'open_interest', 'iv_close'] # iv_close needed
    if not all(c in dft_latest.columns for c in req_cols): return []
    tl = []
    for _, row in dft_latest.iterrows():
        instr = row['instrument_name']; td = ticker_data.get(instr, {}) # Ensure td is a dict
        current_iv = row['iv_close']
        if pd.isna(row['delta']) or pd.isna(row['gamma']) or pd.isna(row['k']) or pd.isna(row['open_interest']) or pd.isna(current_iv) or current_iv <=0: continue
        try: tl.append({"instrument": instr, "strike": int(row['k']), "option_type": row['option_type'], "open_interest": float(row['open_interest']), "delta": float(row['delta']), "gamma": float(row['gamma']), "iv": float(current_iv)})
        except (TypeError, ValueError): continue
    tl.sort(key=lambda x: x['strike'])
    return tl

def plot_gex_by_strike(df_gex_input): # For MM view
    st.subheader("GEX by Strike (Latest Snapshot)")
    if df_gex_input.empty or not all(c in df_gex_input.columns for c in ['k', 'gex', 'option_type']): return
    df_gex = df_gex_input.copy()
    if 'strike' not in df_gex.columns and 'k' in df_gex.columns: df_gex['strike'] = df_gex['k']
    df_plot = df_gex.dropna(subset=['strike', 'gex', 'option_type'])
    if df_plot.empty: return
    try:
        fig = px.bar(df_plot, x="strike", y="gex", color="option_type", title="GEX by Strike", labels={"gex": "GEX Value", "strike": "Strike Price"}, color_discrete_map={'C': 'mediumseagreen', 'P': 'lightcoral'}, barmode='group')
        fig.update_layout(height=400, width=800, bargap=0.1); st.plotly_chart(fig, use_container_width=True)
    except Exception: pass

def plot_net_gex(df_gex_input, spot_price): # For MM view
    st.subheader("Net GEX by Strike (Latest Snapshot)")
    if df_gex_input.empty or not all(c in df_gex_input.columns for c in ['k', 'gex', 'option_type']): return
    df_gex = df_gex_input.copy()
    if 'strike' not in df_gex.columns and 'k' in df_gex.columns: df_gex['strike'] = df_gex['k']
    df_plot_agg = df_gex.dropna(subset=['strike', 'gex', 'option_type'])
    if df_plot_agg.empty: return
    try:
        dfn = df_plot_agg.groupby("strike").apply(lambda x: x.loc[x["option_type"]=="C", "gex"].sum(skipna=True) - x.loc[x["option_type"]=="P", "gex"].sum(skipna=True), include_groups=False).reset_index(name="net_gex")
        if dfn.empty or dfn['net_gex'].isna().all(): return
    except Exception: return
    try:
        dfn["sign"] = dfn["net_gex"].apply(lambda v: "Negative" if v < 0 else "Positive")
        fig = px.bar(dfn, x="strike", y="net_gex", color="sign", color_discrete_map={"Negative": "orange", "Positive": "royalblue"}, title="Net GEX (Calls - Puts)", labels={"net_gex": "Net GEX", "strike": "Strike"})
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        if pd.notna(spot_price): fig.add_vline(x=spot_price, line_dash="dot", line_color="grey", line_width=1, annotation_text=f"Spot {spot_price:.0f}", annotation_position="top right")
        fig.update_layout(height=400, width=800, bargap=0.1); st.plotly_chart(fig, use_container_width=True)
    except Exception: pass

def plot_delta_oi_heatmap_refined(dft_hist, df_spot_hist, expiry_obj):
    st.subheader(f"Historical Delta Exposure Heatmap (Delta * OI)")
    req_hist_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest']; req_spot_cols = ['date_time', 'close']
    if dft_hist.empty or not all(c in dft_hist.columns for c in req_hist_cols) or df_spot_hist.empty or not all(c in df_spot_hist.columns for c in req_spot_cols): return
    try:
        if not (isinstance(dft_hist.index, pd.DatetimeIndex) and dft_hist.index.tz == dt.timezone.utc) and ('date_time' not in dft_hist.columns or dft_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("dft_hist TZ")
        if not (isinstance(df_spot_hist.index, pd.DatetimeIndex) and df_spot_hist.index.tz == dt.timezone.utc) and ('date_time' not in df_spot_hist.columns or df_spot_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("df_spot_hist TZ")
    except Exception: return
    fig = None
    try:
        with st.spinner("Preparing data for Delta OI Heatmap..."):
            dft_hist_local = dft_hist.reset_index() if isinstance(dft_hist.index, pd.DatetimeIndex) else dft_hist.copy(); df_spot_local = df_spot_hist.reset_index() if isinstance(df_spot_hist.index, pd.DatetimeIndex) else df_spot_hist.copy()
            df_merged = merge_spot_to_options(dft_hist_local, df_spot_local, expiry_obj)
            if df_merged.empty: return
            df_merged['delta_hist'] = df_merged.apply(lambda row: compute_delta(row, row['spot_price'], row['date_time']), axis=1)
            df_merged['open_interest'] = pd.to_numeric(df_merged['open_interest'], errors='coerce'); df_merged['delta_hist'] = pd.to_numeric(df_merged['delta_hist'], errors='coerce')
            df_merged = df_merged.dropna(subset=['open_interest', 'delta_hist', 'k'])
            if df_merged.empty: return
            df_merged['delta_oi'] = df_merged['delta_hist'] * df_merged['open_interest']
            df_calls = df_merged[df_merged['option_type'] == 'C'].copy(); df_puts = df_merged[df_merged['option_type'] == 'P'].copy()
            all_strikes_sorted_desc = sorted(df_merged['k'].unique(), reverse=True)
            if not all_strikes_sorted_desc: return
            heatmap_calls_df = pd.DataFrame(); heatmap_puts_df = pd.DataFrame(); heatmap_puts_hover_df = pd.DataFrame()
            if not df_calls.empty: heatmap_calls_df = df_calls.pivot_table(index='k', columns='date_time', values='delta_oi', aggfunc='mean').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
            if not df_puts.empty: df_puts['abs_delta_oi'] = df_puts['delta_oi'].abs(); heatmap_puts_df = df_puts.pivot_table(index='k', columns='date_time', values='abs_delta_oi', aggfunc='mean').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0); heatmap_puts_hover_df = df_puts.pivot_table(index='k', columns='date_time', values='delta_oi', aggfunc='mean').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Call Delta Exposure", "Put Delta Exposure"))
        tick_values = all_strikes_sorted_desc; tick_labels = [str(int(k)) for k in all_strikes_sorted_desc]
        if not heatmap_calls_df.empty:
            positive_call_delta_oi = heatmap_calls_df.values[heatmap_calls_df.values > 0]; cmax_call = np.percentile(positive_call_delta_oi, 98) if positive_call_delta_oi.size > 0 else 1.0
            fig.add_trace(go.Heatmap(z=heatmap_calls_df.values, x=heatmap_calls_df.columns, y=heatmap_calls_df.index, colorscale='Blues', colorbar=dict(title='Call Δ*OI', orientation='h', y=-0.15, x=0.5, len=0.8, xanchor='center', yanchor='top', tickformat=".2f"), zmin=0, zmax=cmax_call, customdata=heatmap_calls_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Call Δ*OI: %{customdata:.2f}<extra></extra>'), row=1, col=1)
        if not heatmap_puts_df.empty:
            positive_put_abs_delta_oi = heatmap_puts_df.values[heatmap_puts_df.values > 0]; cmax_put = np.percentile(positive_put_abs_delta_oi, 98) if positive_put_abs_delta_oi.size > 0 else 1.0
            fig.add_trace(go.Heatmap(z=heatmap_puts_df.values, x=heatmap_puts_df.columns, y=heatmap_puts_df.index, colorscale='Reds', colorbar=dict(title='Abs Put Δ*OI', orientation='h', y=-0.35, x=0.5, len=0.8, xanchor='center', yanchor='top', tickformat=".2f"), zmin=0, zmax=cmax_put, customdata=heatmap_puts_hover_df.values if not heatmap_puts_hover_df.empty else heatmap_puts_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Put Δ*OI: %{customdata:.2f}<extra></extra>'), row=2, col=1)
        fig.update_layout(height=800, hovermode='closest', margin=dict(l=70, r=70, t=80, b=120)); fig.update_xaxes(title_text="Date/Time", row=2, col=1)
        fig.update_yaxes(title_text="Strike Price", tickmode='array', tickvals=tick_values, ticktext=tick_labels, autorange='reversed', row=1, col=1); fig.update_yaxes(title_text="Strike Price", tickmode='array', tickvals=tick_values, ticktext=tick_labels, autorange='reversed', row=2, col=1)
    except Exception: return
    if fig is not None: st.plotly_chart(fig, use_container_width=True)

def plot_gex_heatmap(dft_hist, df_spot_hist, expiry_obj):
    st.subheader(f"Historical Gamma Exposure Heatmap (GEX)")
    req_hist_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest']; req_spot_cols = ['date_time', 'close']
    if dft_hist.empty or not all(c in dft_hist.columns for c in req_hist_cols) or df_spot_hist.empty or not all(c in df_spot_hist.columns for c in req_spot_cols): return
    try:
        if not (isinstance(dft_hist.index, pd.DatetimeIndex) and dft_hist.index.tz == dt.timezone.utc) and ('date_time' not in dft_hist.columns or dft_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("dft_hist TZ")
        if not (isinstance(df_spot_hist.index, pd.DatetimeIndex) and df_spot_hist.index.tz == dt.timezone.utc) and ('date_time' not in df_spot_hist.columns or df_spot_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("df_spot_hist TZ")
    except Exception: return
    fig = None
    try:
        with st.spinner("Preparing data for GEX Heatmap..."):
            dft_hist_local = dft_hist.reset_index() if isinstance(dft_hist.index, pd.DatetimeIndex) else dft_hist.copy(); df_spot_local = df_spot_hist.reset_index() if isinstance(df_spot_hist.index, pd.DatetimeIndex) else df_spot_hist.copy()
            df_merged = merge_spot_to_options(dft_hist_local, df_spot_local, expiry_obj)
            if df_merged.empty: return
            df_merged['gamma'] = df_merged.apply(lambda row: compute_gamma(row, row['spot_price'], row['date_time']), axis=1)
            df_merged['open_interest'] = pd.to_numeric(df_merged['open_interest'], errors='coerce'); df_merged = df_merged.dropna(subset=['gamma'])
            df_merged['gex'] = df_merged.apply(lambda row: compute_gex(row, row['spot_price'], row['open_interest']), axis=1)
            df_merged = df_merged.dropna(subset=['gex', 'k'])
            if df_merged.empty: return
            df_calls = df_merged[df_merged['option_type'] == 'C'].copy(); df_puts = df_merged[df_merged['option_type'] == 'P'].copy()
            all_strikes_sorted_desc = sorted(df_merged['k'].unique(), reverse=True)
            if not all_strikes_sorted_desc: return
            heatmap_gex_calls_df = pd.DataFrame(); heatmap_gex_puts_df = pd.DataFrame()
            if not df_calls.empty: heatmap_gex_calls_df = df_calls.pivot_table(index='k', columns='date_time', values='gex', aggfunc='mean').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
            if not df_puts.empty: heatmap_gex_puts_df = df_puts.pivot_table(index='k', columns='date_time', values='gex', aggfunc='mean').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Call GEX Heatmap", "Put GEX Heatmap"))
        tick_values = all_strikes_sorted_desc; tick_labels = [str(int(k)) for k in all_strikes_sorted_desc]
        if not heatmap_gex_calls_df.empty:
            positive_call_gex = heatmap_gex_calls_df.values[heatmap_gex_calls_df.values > 0]; cmax_gex_call = np.percentile(positive_call_gex, 98) if positive_call_gex.size > 0 else 1.0
            fig.add_trace(go.Heatmap(z=heatmap_gex_calls_df.values, x=heatmap_gex_calls_df.columns, y=heatmap_gex_calls_df.index, colorscale='Blues', colorbar=dict(title='Call GEX', orientation='h', y=-0.15, x=0.5, len=0.8, xanchor='center', yanchor='top', tickformat=",.1f"), zmin=0, zmax=cmax_gex_call, customdata=heatmap_gex_calls_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Call GEX: %{customdata:,.2f}<extra></extra>'), row=1, col=1)
        if not heatmap_gex_puts_df.empty:
            positive_put_gex = heatmap_gex_puts_df.values[heatmap_gex_puts_df.values > 0]; cmax_gex_put = np.percentile(positive_put_gex, 98) if positive_put_gex.size > 0 else 1.0
            fig.add_trace(go.Heatmap(z=heatmap_gex_puts_df.values, x=heatmap_gex_puts_df.columns, y=heatmap_gex_puts_df.index, colorscale='Reds', colorbar=dict(title='Put GEX', orientation='h', y=-0.35, x=0.5, len=0.8, xanchor='center', yanchor='top', tickformat=",.1f"), zmin=0, zmax=cmax_gex_put, customdata=heatmap_gex_puts_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Put GEX: %{customdata:,.2f}<extra></extra>'), row=2, col=1)
        fig.update_layout(height=800, hovermode='closest', margin=dict(l=70, r=70, t=80, b=120)); fig.update_xaxes(title_text="Date/Time", row=2, col=1)
        fig.update_yaxes(title_text="Strike Price", tickmode='array', tickvals=tick_values, ticktext=tick_labels, autorange='reversed', row=1, col=1); fig.update_yaxes(title_text="Strike Price", tickmode='array', tickvals=tick_values, ticktext=tick_labels, autorange='reversed', row=2, col=1)
    except Exception: return
    if fig is not None: st.plotly_chart(fig, use_container_width=True)

def plot_net_delta_flow_heatmap(dft_hist, df_spot_hist, expiry_obj, coin):
    st.subheader(f"Net Delta Flow Heatmap (Δ(Delta * OI))"); st.caption("Blue = Net Delta Buying | Red = Net Delta Selling")
    req_hist_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest']; req_spot_cols = ['date_time', 'close']
    if dft_hist.empty or not all(c in dft_hist.columns for c in req_hist_cols) or df_spot_hist.empty or not all(c in df_spot_hist.columns for c in req_spot_cols): return
    try:
        if not (isinstance(dft_hist.index, pd.DatetimeIndex) and dft_hist.index.tz == dt.timezone.utc) and ('date_time' not in dft_hist.columns or dft_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("dft_hist TZ")
        if not (isinstance(df_spot_hist.index, pd.DatetimeIndex) and df_spot_hist.index.tz == dt.timezone.utc) and ('date_time' not in df_spot_hist.columns or df_spot_hist['date_time'].dt.tz != dt.timezone.utc): raise ValueError("df_spot_hist TZ")
    except Exception: return
    fig = None
    try:
        with st.spinner("Preparing data for Net Delta Flow Heatmap..."):
            dft_hist_local = dft_hist.reset_index() if isinstance(dft_hist.index, pd.DatetimeIndex) else dft_hist.copy(); df_spot_local = df_spot_hist.reset_index() if isinstance(df_spot_hist.index, pd.DatetimeIndex) else df_spot_hist.copy()
            df_merged = merge_spot_to_options(dft_hist_local, df_spot_local, expiry_obj)
            if df_merged.empty: return
            df_merged['delta_hist'] = df_merged.apply(lambda row: compute_delta(row, row['spot_price'], row['date_time']), axis=1)
            df_merged['open_interest'] = pd.to_numeric(df_merged['open_interest'], errors='coerce'); df_merged['delta_hist'] = pd.to_numeric(df_merged['delta_hist'], errors='coerce')
            df_merged = df_merged.dropna(subset=['open_interest', 'delta_hist'])
            if df_merged.empty: return
            df_merged['delta_oi'] = df_merged['delta_hist'] * df_merged['open_interest']
            df_merged = df_merged.sort_values(by=['instrument_name', 'date_time'])
            df_merged['delta_oi_instrument_flow'] = df_merged.groupby('instrument_name')['delta_oi'].diff().fillna(0)
            df_flow_agg = df_merged.groupby(['date_time', 'k', 'option_type'])['delta_oi_instrument_flow'].sum().reset_index().rename(columns={'delta_oi_instrument_flow': 'net_delta_flow'})
            df_calls_flow = df_flow_agg[df_flow_agg['option_type'] == 'C'].copy(); df_puts_flow = df_flow_agg[df_flow_agg['option_type'] == 'P'].copy()
            heatmap_calls_df = pd.DataFrame(); heatmap_puts_df = pd.DataFrame()
            all_strikes_sorted_desc = sorted(df_flow_agg['k'].unique(), reverse=True)
            if not df_calls_flow.empty: heatmap_calls_df = df_calls_flow.pivot_table(index='k', columns='date_time', values='net_delta_flow', aggfunc='sum').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
            if not df_puts_flow.empty: heatmap_puts_df = df_puts_flow.pivot_table(index='k', columns='date_time', values='net_delta_flow', aggfunc='sum').fillna(0).reindex(all_strikes_sorted_desc, fill_value=0)
        df_spot_plot = df_spot_hist[['date_time', 'close']].copy(); df_spot_plot['close'] = pd.to_numeric(df_spot_plot['close'], errors='coerce'); df_spot_plot = df_spot_plot.dropna(subset=['close']).set_index('date_time').sort_index()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Call Net Delta Flow vs Price", "Put Net Delta Flow vs Price"), specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
        combined_flow = pd.concat([heatmap_calls_df, heatmap_puts_df]) if not heatmap_calls_df.empty or not heatmap_puts_df.empty else pd.DataFrame(); non_zero_flow = combined_flow.values[combined_flow.values != 0]
        limit = np.percentile(np.abs(non_zero_flow), 98) if non_zero_flow.size > 0 else 1.0; limit = max(limit, 0.01); color_range = [-limit, limit]
        if not heatmap_calls_df.empty: fig.add_trace(go.Heatmap(z=heatmap_calls_df.values, x=heatmap_calls_df.columns, y=heatmap_calls_df.index.astype(str), colorscale='RdBu', zmid=0, zmin=color_range[0], zmax=color_range[1], colorbar=dict(title='Call Net Δ Flow', len=0.45, y=0.8, tickformat=".2f"), hoverongaps=False, name="Call Flow", showlegend=False, customdata=heatmap_calls_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Net Delta Flow: %{customdata:.2f}<extra></extra>'), secondary_y=False, row=1, col=1)
        if not heatmap_puts_df.empty: fig.add_trace(go.Heatmap(z=heatmap_puts_df.values, x=heatmap_puts_df.columns, y=heatmap_puts_df.index.astype(str), colorscale='RdBu', zmid=0, zmin=color_range[0], zmax=color_range[1], colorbar=dict(title='Put Net Δ Flow', len=0.45, y=0.2, tickformat=".2f"), hoverongaps=False, name="Put Flow", showlegend=False, customdata=heatmap_puts_df.values, hovertemplate='<b>Strike: %{y}</b><br>Time: %{x}<br>Net Delta Flow: %{customdata:.2f}<extra></extra>'), secondary_y=False, row=2, col=1)
        if not df_spot_plot.empty:
            x_min_hm = heatmap_calls_df.columns.min() if not heatmap_calls_df.empty else (heatmap_puts_df.columns.min() if not heatmap_puts_df.empty else None); x_max_hm = heatmap_calls_df.columns.max() if not heatmap_calls_df.empty else (heatmap_puts_df.columns.max() if not heatmap_puts_df.empty else None)
            spot_plot_filtered = df_spot_plot;
            if x_min_hm and x_max_hm: spot_plot_filtered = df_spot_plot[(df_spot_plot.index >= x_min_hm) & (df_spot_plot.index <= x_max_hm)]
            if not spot_plot_filtered.empty:
                fig.add_trace(go.Scatter(x=spot_plot_filtered.index, y=spot_plot_filtered['close'], mode='lines', name=f'{coin} Price', line=dict(color='rgba(211, 211, 211, 0.7)', width=1.5), showlegend=True), secondary_y=True, row=1, col=1)
                fig.add_trace(go.Scatter(x=spot_plot_filtered.index, y=spot_plot_filtered['close'], mode='lines', name=f'{coin} Price', line=dict(color='rgba(211, 211, 211, 0.7)', width=1.5), showlegend=False), secondary_y=True, row=2, col=1)
        fig.update_layout(height=750, hovermode='closest', legend_title="Metrics", legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5), plot_bgcolor='rgba(17, 17, 17, 1)', paper_bgcolor='rgba(17, 17, 17, 1)', font=dict(color='lightgrey'), margin=dict(l=70, r=70, t=80, b=80), yaxis=dict(title="Strike Price", side='left', categoryorder='array', categoryarray=all_strikes_sorted_desc), yaxis2=dict(title="Price ($)", side='right', overlaying='y', showgrid=False), yaxis3=dict(title="Strike Price", side='left', categoryorder='array', categoryarray=all_strikes_sorted_desc), yaxis4=dict(title="Price ($)", side='right', overlaying='y3', showgrid=False))
        x_axis_range = [heatmap_calls_df.columns.min(), heatmap_calls_df.columns.max()] if not heatmap_calls_df.empty else None
        fig.update_xaxes(range=x_axis_range, row=1, col=1); fig.update_xaxes(title_text="Date/Time", range=x_axis_range, row=2, col=1, gridcolor='rgba(68, 68, 68, 0.5)')
        fig.update_yaxes(tickmode='auto', nticks=15, row=1, col=1, secondary_y=False); fig.update_yaxes(tickmode='auto', nticks=15, row=2, col=1, secondary_y=False)
    except Exception: return
    if fig is not None: st.plotly_chart(fig, use_container_width=True)

# --- Functions for Premium Bias Plot ---
def calculate_atm_premium_data(dft, df_krak_5m, selected_expiry_obj):
    expiry_label = selected_expiry_obj.strftime('%d%b%y') if isinstance(selected_expiry_obj, dt.datetime) else "N/A"
    if dft.empty or df_krak_5m.empty or not all(c in dft.columns for c in ['date_time', 'k', 'mark_price_close', 'option_type']) or not all(c in df_krak_5m.columns for c in ['date_time', 'close']): return pd.DataFrame()
    try:
        dft_local = dft.copy(); spot_local = df_krak_5m.copy(); dft_local = dft_local.sort_values('date_time'); spot_local = spot_local.sort_values('date_time')
        df_merged = pd.merge_asof(dft_local, spot_local[['date_time', 'close']].rename(columns={'close': 'spot_price'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
        df_merged = df_merged.dropna(subset=['date_time', 'spot_price', 'k', 'mark_price_close', 'option_type'])
        if df_merged.empty: return pd.DataFrame()
        for col in ['spot_price', 'k', 'mark_price_close']: df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        df_merged = df_merged.dropna(subset=['spot_price', 'k', 'mark_price_close'])
        if df_merged.empty: return pd.DataFrame()
        atm_premiums = []
        for ts in sorted(df_merged['date_time'].unique()):
            df_ts = df_merged[df_merged['date_time'] == ts]
            if df_ts.empty: continue
            spot = df_ts['spot_price'].iloc[0];
            if pd.isna(spot): continue
            try: atm_idx = abs(df_ts['k'] - spot).idxmin(); atm_strike = df_ts.loc[atm_idx, 'k']
            except ValueError: continue
            call_price_atm = df_ts[(df_ts['k'] == atm_strike) & (df_ts['option_type'] == 'C')]['mark_price_close'].iloc[0] if not df_ts[(df_ts['k'] == atm_strike) & (df_ts['option_type'] == 'C')].empty else np.nan
            put_price_atm = df_ts[(df_ts['k'] == atm_strike) & (df_ts['option_type'] == 'P')]['mark_price_close'].iloc[0] if not df_ts[(df_ts['k'] == atm_strike) & (df_ts['option_type'] == 'P')].empty else np.nan
            difference = call_price_atm - put_price_atm if pd.notna(call_price_atm) and pd.notna(put_price_atm) else np.nan
            atm_premiums.append({'date_time': ts, 'atm_call_premium': call_price_atm, 'atm_put_premium': put_price_atm, 'atm_difference': difference, 'atm_strike': atm_strike, 'spot_price': spot})
        return pd.DataFrame(atm_premiums)
    except Exception: return pd.DataFrame()

def calculate_itm_premium_data(dft, df_krak_5m, selected_expiry_obj):
    expiry_label = selected_expiry_obj.strftime('%d%b%y') if isinstance(selected_expiry_obj, dt.datetime) else "N/A"
    if dft.empty or df_krak_5m.empty or not all(c in dft.columns for c in ['date_time', 'k', 'mark_price_close', 'option_type']) or not all(c in df_krak_5m.columns for c in ['date_time', 'close']): return pd.DataFrame()
    try:
        dft_local = dft.copy(); spot_local = df_krak_5m.copy(); dft_local = dft_local.sort_values('date_time'); spot_local = spot_local.sort_values('date_time')
        df_merged = pd.merge_asof(dft_local, spot_local[['date_time', 'close']].rename(columns={'close': 'spot_price'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
        df_merged = df_merged.dropna(subset=['date_time', 'spot_price', 'k', 'mark_price_close', 'option_type'])
        if df_merged.empty: return pd.DataFrame()
        for col in ['spot_price', 'k', 'mark_price_close']: df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        df_merged = df_merged.dropna(subset=['spot_price', 'k', 'mark_price_close'])
        if df_merged.empty: return pd.DataFrame()
        itm_premiums = []
        for ts in sorted(df_merged['date_time'].unique()):
            df_ts = df_merged[df_merged['date_time'] == ts]
            if df_ts.empty: continue
            spot = df_ts['spot_price'].iloc[0];
            if pd.isna(spot): continue
            itm_calls = df_ts[(df_ts['option_type'] == 'C') & (df_ts['k'] < spot)].sort_values('k', ascending=False)
            nearest_itm_call_data = itm_calls.iloc[0] if not itm_calls.empty else None
            itm_puts = df_ts[(df_ts['option_type'] == 'P') & (df_ts['k'] > spot)].sort_values('k', ascending=True)
            nearest_itm_put_data = itm_puts.iloc[0] if not itm_puts.empty else None
            call_price_itm = nearest_itm_call_data['mark_price_close'] if nearest_itm_call_data is not None else np.nan
            put_price_itm = nearest_itm_put_data['mark_price_close'] if nearest_itm_put_data is not None else np.nan
            difference = call_price_itm - put_price_itm if pd.notna(call_price_itm) and pd.notna(put_price_itm) else np.nan
            itm_premiums.append({'date_time': ts, 'itm_call_premium': call_price_itm, 'itm_put_premium': put_price_itm, 'itm_difference': difference}) # No need for strikes/spot here
        return pd.DataFrame(itm_premiums)
    except Exception: return pd.DataFrame()
def calculate_hurst_lo_modified(series, min_n=10, max_n=None, q_method='auto'): # q_method kept for signature consistency if needed elsewhere
    """
    Calculates the Hurst exponent using Lo's (1991) modified R/S analysis,
    which is more robust to short-term dependence.
    NOTE: For this focused version, the q-adjustment for autocovariance is simplified
    and effectively defaults to classic R/S if calculate_lo_modified_variance isn't fully implemented.
    """
    if isinstance(series, list): series = np.array(series)
    elif isinstance(series, pd.Series): series = series.values

    series = series[~np.isnan(series)]; N = len(series)

    if max_n is None: max_n = N // 2
    max_n = min(max_n, N - 1); min_n = max(2, min_n) # n must be at least 2 and < N

    if N < 20 or min_n >= max_n : # Need at least a few points and valid range
        logging.warning(f"Series too short (N={N}) or invalid n range ({min_n}-{max_n}) for Hurst (Lo) calculation.")
        return np.nan, pd.DataFrame()

    # Generate intervals (logarithmic spacing is generally preferred)
    ns = np.unique(np.geomspace(min_n, max_n, num=20, dtype=int))
    ns = [n_val for n_val in ns if n_val >= min_n] # Ensure min_n is respected

    if not ns:
         logging.warning("No valid window sizes 'n' found for Hurst (Lo).")
         return np.nan, pd.DataFrame()

    rs_values = []; valid_ns = []

    for n_interval in ns: # Renamed n to n_interval to avoid conflict if Lo's n is used later
        # Determine q for this n_interval (simplified for this version)
        q = 0 # Effectively classic R/S if calculate_lo_modified_variance is simplified
        if isinstance(q_method, int): q = max(0, min(q_method, n_interval - 1))
        elif q_method == 'auto' and n_interval > 10 : q = max(0, min(int(np.floor(1.1447 * (n_interval**(1/3)))), n_interval - 1))

        rs_chunk_list = [] # Renamed rs_chunk to rs_chunk_list
        num_chunks = N // n_interval
        if num_chunks < 1: continue

        for i in range(num_chunks):
            chunk = series[i * n_interval : (i + 1) * n_interval]
            if len(chunk) < 2: continue # Need at least 2 points for std dev

            mean = np.mean(chunk)
            if pd.isna(mean): continue # Skip if mean cannot be calculated (e.g., all NaNs)

            if np.allclose(chunk, mean): continue # Skip constant chunks (R=0, S=0)

            mean_adjusted = chunk - mean
            cum_dev = np.cumsum(mean_adjusted)
            if cum_dev.size == 0: continue # Should not happen if chunk had len >=2 and not constant

            # R calculation: max of cum_dev - min of cum_dev (including initial zero)
            cum_dev_with_zero = np.insert(cum_dev, 0, 0.0)
            R = np.ptp(cum_dev_with_zero) # Peak-to-peak is max - min

            if pd.isna(R) or R < 0: continue # Range must be non-negative

            # S calculation (Standard Deviation)
            # For Lo's R/S_q, S_q is the square root of the modified variance.
            # Here, we'll use standard deviation for classic R/S as a fallback/simplification.
            # The full calculate_lo_modified_variance is needed for true Lo's R/S_q.
            # S = np.std(chunk, ddof=0) # Population standard deviation as per original R/S
            # If calculate_lo_modified_variance IS NOT defined or used:
            S = np.std(chunk, ddof=0) # Use ddof=0 for population std dev as in classic R/S

            if pd.isna(S) or S <= 1e-9: continue # S must be positive

            rs_ratio = R / S
            if not pd.isna(rs_ratio) and rs_ratio >= 0: # R/S must be non-negative
                rs_chunk_list.append(rs_ratio)

        if rs_chunk_list: # If we got valid R/S values for this interval size
            rs_values.append(np.mean(rs_chunk_list))
            valid_ns.append(n_interval)
    
    if len(valid_ns) < 3: # Need at least 3 points for a reliable regression
        logging.warning(f"Insufficient valid R/S points ({len(valid_ns)}) for Hurst regression.")
        return np.nan, pd.DataFrame()

    results_df = pd.DataFrame({'interval': valid_ns, 'rs_mean': rs_values})
    results_df = results_df.replace([np.inf, -np.inf], np.nan).dropna() # Clean up final results
    if len(results_df) < 3:
        logging.warning(f"Insufficient valid R/S points ({len(results_df)}) after final dropna for Hurst regression.")
        return np.nan, pd.DataFrame()

    log_intervals = np.log(results_df['interval'])
    log_rs = np.log(results_df['rs_mean']) # Ensure rs_mean is positive here

    try:
        hurst, _, _, _, _ = linregress(log_intervals, log_rs) # Using scipy.stats.linregress
        return hurst, results_df
    except (np.linalg.LinAlgError, ValueError) as e:
        logging.error(f"Error during Hurst (Lo) polyfit: {e}")
        return np.nan, results_df
    except Exception as e:
        logging.error(f"Unexpected error during Hurst (Lo) polyfit: {e}")
        return np.nan, results_df
def plot_hurst_exponent(hurst_val, hurst_data_df):
    if pd.isna(hurst_val) or hurst_data_df.empty or not all(c in hurst_data_df.columns for c in ['interval', 'rs_mean']): return
    plot_data = hurst_data_df[hurst_data_df['rs_mean'] > 1e-12].copy()
    if plot_data.empty or len(plot_data) < 3: return
    log_intervals = np.log(plot_data['interval']); log_rs = np.log(plot_data['rs_mean'])
    fitted_rs = None; label_h = hurst_val
    try:
        if len(plot_data) >= 3:
            hurst_plot, intercept_plot, _, _, _ = linregress(log_intervals, log_rs)
            fitted_rs = intercept_plot + hurst_plot * log_intervals
    except Exception: pass
    fig_hurst = go.Figure()
    fig_hurst.add_trace(go.Scatter(x=log_intervals, y=log_rs, mode='markers', name='Log(Avg R/S) vs Log(n)', marker=dict(color='blue')))
    if fitted_rs is not None: fig_hurst.add_trace(go.Scatter(x=log_intervals, y=fitted_rs, mode='lines', name=f'Fit (H={label_h:.3f})', line=dict(color='red', dash='dash')))
    fig_hurst.update_layout(title="Hurst Exponent (Classic R/S)", xaxis_title="Log(Time Interval n)", yaxis_title="Log(Mean R/S)", height=400, width=800)
    st.plotly_chart(fig_hurst, use_container_width=True)
    regime = "Trending (H > 0.5)" if 0.55 < hurst_val <= 1.0 else "Mean-Reverting (H < 0.5)" if 0.0 <= hurst_val < 0.45 else "Random Walk (H ≈ 0.5)" if 0.45 <= hurst_val <= 0.55 else f"Unusual (H={hurst_val:.3f})"
    st.write(f"**Hurst Exponent (Classic R/S, Daily Log Returns): {hurst_val:.3f}** | **Implied Regime:** {regime}")

def calculate_and_display_autocorrelation(daily_log_returns, windows=[7, 15, 30], threshold=0.05):
    st.subheader("Implied Market Regime (Daily Autocorrelation)")
    if not isinstance(daily_log_returns, pd.Series) or daily_log_returns.empty: return
    cleaned_returns = daily_log_returns.dropna()
    if cleaned_returns.empty: return
    regime_map = {"≈": "Mean Reverting", "—": "Random Walk", "↑": "Persistent Up", "↓": "Persistent Down", "?": "Unknown", "!": "Error"}
    cols = st.columns(len(windows))
    for i, window in enumerate(windows):
        with cols[i]:
            regime_symbol = "?"
            if len(cleaned_returns) >= window and len(cleaned_returns.tail(window)) >= 2:
                try:
                    autocorr_val = cleaned_returns.tail(window).autocorr(lag=1)
                    if pd.isna(autocorr_val): pass
                    elif autocorr_val < -threshold: regime_symbol = "≈"
                    elif autocorr_val > threshold:
                        avg_return = cleaned_returns.tail(window).mean()
                        if pd.notna(avg_return): regime_symbol = "↑" if avg_return > 0 else "↓"
                    else: regime_symbol = "—"
                except Exception: regime_symbol = "!"
            st.markdown(f"<p style='text-align: center; font-size: small; color: grey; margin-bottom: -10px;'>{window}-Day</p>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; margin-bottom: -10px;'>{regime_symbol}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: small; margin-top: 0px;'>{regime_map.get(regime_symbol, 'Unknown')}</p>", unsafe_allow_html=True)
def display_mm_gamma_adjustment_analysis(dft_latest_snap, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    """
    Displays an indicative Delta-Gamma hedge adjustment for an assumed Market Maker (MM) book.
    Assumes MM is short all options in dft_latest_snap for the selected expiry.
    Selects a near-ATM call from the same expiry as the gamma hedging instrument.
    """
    st.subheader("MM Indicative Delta-Gamma Hedge Adjustment (Selected Expiry)")
    st.caption("Assumes Market Maker is short the entire displayed option book for this expiry. "
               "Shows theoretical adjustment using a near-ATM call from the same expiry to hedge gamma.")

    log_prefix = "MM_DG_Adjust:"

    # --- 1. Input Validation ---
    required_cols = ['instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'expiry_datetime_col'] # Added expiry_datetime_col
    if dft_latest_snap.empty or not all(c in dft_latest_snap.columns for c in required_cols):
        st.warning(f"{log_prefix} Cannot perform analysis: Latest snapshot data missing required columns or empty.")
        logging.warning(f"{log_prefix} dft_latest_snap empty or missing columns.")
        return
    if pd.isna(spot_price) or spot_price <= 0:
        st.warning(f"{log_prefix} Invalid spot price: {spot_price}.")
        return

    df_book = dft_latest_snap.copy()
    df_book['open_interest'] = pd.to_numeric(df_book['open_interest'], errors='coerce').fillna(0)
    df_book = df_book[df_book['open_interest'] > 0] # Consider only options with OI

    if df_book.empty:
        st.info(f"{log_prefix} No options with open interest in the latest snapshot for this expiry.")
        return

    # --- 2. MM Book Greeks (Pre-Hedge) ---
    # Assuming MM is SHORT the book, so multiply greeks by -1 * open_interest
    st.write("Calculating MM Book Greeks (assuming MM is short the book)...") # For user feedback
    df_book['mm_delta_pos'] = -1 * df_book.apply(
        lambda r: compute_delta(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1
    ) * df_book['open_interest']

    df_book['mm_gamma_pos'] = -1 * df_book.apply(
        lambda r: compute_gamma(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1
    ) * df_book['open_interest']

    mm_net_delta_initial = df_book['mm_delta_pos'].sum(skipna=True)
    mm_net_gamma_initial = df_book['mm_gamma_pos'].sum(skipna=True)

    if pd.isna(mm_net_delta_initial) or pd.isna(mm_net_gamma_initial):
        st.error(f"{log_prefix} Failed to calculate initial MM book greeks. Check option data.")
        return

    st.metric("MM Initial Net Delta (Book)", f"{mm_net_delta_initial:,.2f}")
    st.metric("MM Initial Net Gamma (Book)", f"{mm_net_gamma_initial:,.4f}")

    # --- 3. Select Gamma Hedging Instrument (Near-ATM Call from same expiry) ---
    gamma_hedger_selected = None
    all_calls_in_book = df_book[df_book['option_type'] == 'C'].copy()
    if not all_calls_in_book.empty:
        all_calls_in_book['moneyness_dist'] = abs(all_calls_in_book['k'] - spot_price)
        # Prefer slightly OTM or ATM calls for gamma hedging
        atm_ish_calls = all_calls_in_book[all_calls_in_book['k'] >= spot_price].sort_values('moneyness_dist')
        if not atm_ish_calls.empty:
            gamma_hedger_selected_name = atm_ish_calls['instrument_name'].iloc[0]
        else: # Fallback to closest ITM call if no ATM/OTM
            atm_ish_calls = all_calls_in_book.sort_values('moneyness_dist')
            if not atm_ish_calls.empty:
                gamma_hedger_selected_name = atm_ish_calls['instrument_name'].iloc[0]
            else: gamma_hedger_selected_name = None
        
        if gamma_hedger_selected_name:
            # Get the full row for the selected hedger from the original df_book to ensure all columns are present
            gamma_hedger_selected_row_data = df_book[df_book['instrument_name'] == gamma_hedger_selected_name].iloc[[0]]
            if not gamma_hedger_selected_row_data.empty:
                 gamma_hedger_selected = gamma_hedger_selected_row_data.iloc[0].copy() # as a Series
    
    if gamma_hedger_selected is None:
        st.warning(f"{log_prefix} Could not select a suitable Call option from the book for gamma hedging.")
        return

    hedger_details_name = gamma_hedger_selected['instrument_name']
    st.info(f"Selected Gamma Hedging Instrument: {hedger_details_name}")

    # Calculate Greeks for ONE unit of the hedging instrument
    D_h = compute_delta(gamma_hedger_selected, spot_price, snapshot_time_utc, risk_free_rate)
    G_h = compute_gamma(gamma_hedger_selected, spot_price, snapshot_time_utc, risk_free_rate)

    if pd.isna(D_h) or pd.isna(G_h) or abs(G_h) < 1e-7: # Gamma of hedger must be significant
        st.error(f"{log_prefix} Invalid greeks for selected gamma hedging instrument ({hedger_details_name}): Delta={D_h}, Gamma={G_h}")
        return

    # --- 4. Calculate Gamma Hedge Quantity (N_h) ---
    N_h = -mm_net_gamma_initial / G_h

    # --- 5. Calculate Delta Impact of Gamma Hedge ---
    delta_from_gamma_hedge = N_h * D_h

    # --- 6. Calculate New Net Delta (Post-Gamma Hedge) ---
    mm_net_delta_post_gamma_hedge = mm_net_delta_initial + delta_from_gamma_hedge

    # --- 7. Calculate Final Underlying Hedge ---
    underlying_hedge_qty = -mm_net_delta_post_gamma_hedge

    # --- 8. Display Results ---
    st.markdown("#### Indicative Gamma Hedge Adjustment:")
    cols_gamma_hedge = st.columns(3)
    with cols_gamma_hedge[0]:
        st.metric("Gamma Hedger Delta (Dₕ)", f"{D_h:.4f}")
    with cols_gamma_hedge[1]:
        st.metric("Gamma Hedger Gamma (Gₕ)", f"{G_h:.6f}")
    with cols_gamma_hedge[2]:
        action_gh = "Buy" if N_h > 0 else "Sell" if N_h < 0 else "Hold"
        st.metric(f"Hedge Option Qty ({action_gh})", f"{abs(N_h):,.2f} units")
    
    st.metric("Delta Change from Gamma Hedge", f"{delta_from_gamma_hedge:,.2f}")
    
    st.markdown("---")
    st.markdown("#### Indicative Final Delta Hedge (Post-Gamma Adj.):")
    st.metric("MM Net Delta (After Gamma Hedge)", f"{mm_net_delta_post_gamma_hedge:,.2f}")
    
    action_underlying = "Buy" if underlying_hedge_qty > 0 else "Sell" if underlying_hedge_qty < 0 else "Hold"
    st.metric(f"Final Underlying Hedge ({action_underlying} Spot/Perp)", f"{abs(underlying_hedge_qty):,.2f} {st.session_state.selected_coin}")

    final_net_delta_book = mm_net_delta_post_gamma_hedge + underlying_hedge_qty
    st.success(f"**Resulting Book Net Delta (Post-All Adjustments):** {final_net_delta_book:,.4f} (should be ~0)")
    st.caption("This indicates the spot/perp hedge needed for the MM to become delta-neutral *after* also neutralizing gamma with the chosen option.")
    logging.info(f"{log_prefix} Displayed MM gamma adjustment analysis.")            
def plot_combined_premium_difference(df_atm_results, df_itm_results, expiry_label): # df_agg_oi_results removed as it's not used
    st.subheader(f"Options Premium Bias Comparison (Expiry: {expiry_label})")
    valid_atm = isinstance(df_atm_results, pd.DataFrame) and not df_atm_results.empty and 'atm_difference' in df_atm_results.columns and 'date_time' in df_atm_results.columns
    valid_itm = isinstance(df_itm_results, pd.DataFrame) and not df_itm_results.empty and 'itm_difference' in df_itm_results.columns and 'date_time' in df_itm_results.columns
    if not (valid_atm or valid_itm): return # Simplified check
    dfs_to_merge = []
    if valid_atm: dfs_to_merge.append(df_atm_results[['date_time', 'atm_difference']].set_index('date_time'))
    if valid_itm: dfs_to_merge.append(df_itm_results[['date_time', 'itm_difference']].set_index('date_time'))
    if not dfs_to_merge: return
    df_combined_diff = pd.concat(dfs_to_merge, axis=1, join='outer').sort_index()
    diff_cols = [col for col in ['atm_difference', 'itm_difference'] if col in df_combined_diff.columns] # Removed Agg_OI
    df_combined_diff = df_combined_diff.dropna(subset=diff_cols, how='all').reset_index()
    if df_combined_diff.empty or len(df_combined_diff) < 2: return
    plot_range = None
    try:
        min_plot_time = df_combined_diff['date_time'].min(); max_plot_time = df_combined_diff['date_time'].max()
        time_range_seconds = (max_plot_time - min_plot_time).total_seconds()
        padding_seconds = max(300, time_range_seconds * 0.02)
        plot_range = [min_plot_time - pd.Timedelta(seconds=padding_seconds), max_plot_time + pd.Timedelta(seconds=padding_seconds)]
    except Exception: pass
    try:
        fig = go.Figure()
        if 'atm_difference' in df_combined_diff.columns: fig.add_trace(go.Scatter(x=df_combined_diff['date_time'], y=df_combined_diff['atm_difference'], mode='lines', name='ATM Diff (Call - Put)', line=dict(color='royalblue', width=2), connectgaps=False))
        if 'itm_difference' in df_combined_diff.columns: fig.add_trace(go.Scatter(x=df_combined_diff['date_time'], y=df_combined_diff['itm_difference'], mode='lines', name='Nearest ITM Diff (Call - Put)', line=dict(color='darkorange', width=2, dash='dash'), connectgaps=False))
        fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1.5, annotation_text="Call Bias / Put Bias", annotation_position="bottom right")
        fig.update_layout(title=f"Options Premium Bias (Expiry: {expiry_label})", xaxis_title="Date/Time", yaxis_title="Premium Difference ($)", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified')
        fig.update_xaxes(tickformat="%m/%d %H:%M", range=plot_range)
        st.plotly_chart(fig, use_container_width=True)
    except Exception: pass
class MatrixDeltaGammaHedgeSimple:
    def __init__(self, df_portfolio_options, spot_df, symbol="BTC", risk_free_rate=0.0,
                 gamma_hedge_instrument_details=None):
        self.df_portfolio_options = df_portfolio_options.copy()
        self.spot_df = spot_df.copy()
        self.symbol = symbol.upper()
        self.risk_free_rate = risk_free_rate
        self.gamma_hedge_instrument_details = gamma_hedge_instrument_details
        self.portfolio_state_log = []
        self.hedge_actions_log = []
        self.current_underlying_hedge_qty = 0.0
        self.current_gamma_option_hedge_qty = 0.0
        self._validate_inputs()
        logging.info(f"MatrixDeltaGammaHedgeSimple initialized for {self.symbol}. RF_Rate={self.risk_free_rate}")
        if self.gamma_hedge_instrument_details:
            logging.info(f"Gamma hedging with: {self.gamma_hedge_instrument_details.get('name', 'N/A')}")
        else:
            logging.warning("MatrixDeltaGammaHedgeSimple: No gamma_hedge_instrument_details provided. Gamma hedging will not be active.")

    def _validate_inputs(self):
        if self.symbol not in ["BTC", "ETH"]: raise ValueError(f"Incorrect symbol: {self.symbol}")
        req_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'mark_price_close', 'expiry_datetime_col']
        if self.df_portfolio_options.empty or not all(c in self.df_portfolio_options.columns for c in req_cols):
            raise ValueError(f"df_portfolio_options missing columns: {[c for c in req_cols if c not in self.df_portfolio_options.columns]}")
        for df in [self.df_portfolio_options, self.spot_df]:
            if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
                df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
            elif df['date_time'].dt.tz is None: df['date_time'] = df['date_time'].dt.tz_localize('UTC')
            elif df['date_time'].dt.tz != dt.timezone.utc: df['date_time'] = df['date_time'].dt.tz_convert('UTC')
        if not pd.api.types.is_datetime64_any_dtype(self.df_portfolio_options['expiry_datetime_col']):
            self.df_portfolio_options['expiry_datetime_col'] = pd.to_datetime(self.df_portfolio_options['expiry_datetime_col'], utc=True)

    def _get_portfolio_greeks(self, timestamp, spot_price):
        if self.df_portfolio_options.empty: return 0.0, 0.0, 0.0
        options_at_ts = self.df_portfolio_options[self.df_portfolio_options['date_time'] == timestamp].copy()
        if options_at_ts.empty: return np.nan, np.nan, np.nan
        options_at_ts = options_at_ts[options_at_ts['expiry_datetime_col'] > timestamp]
        if options_at_ts.empty: return 0.0, 0.0, 0.0
        options_at_ts['value_pos'] = options_at_ts['mark_price_close'] * options_at_ts['open_interest']
        options_at_ts['delta_pos'] = options_at_ts.apply(lambda r: compute_delta(r, spot_price, timestamp, self.risk_free_rate), axis=1) * options_at_ts['open_interest']
        options_at_ts['gamma_pos'] = options_at_ts.apply(lambda r: compute_gamma(r, spot_price, timestamp, self.risk_free_rate), axis=1) * options_at_ts['open_interest']
        return options_at_ts['value_pos'].sum(skipna=True), options_at_ts['delta_pos'].sum(skipna=True), options_at_ts['gamma_pos'].sum(skipna=True)

    def _get_gamma_hedger_greeks_and_price(self, timestamp, spot_price):
        if not self.gamma_hedge_instrument_details:
            return np.nan, np.nan, np.nan

        details = self.gamma_hedge_instrument_details
        hedger_name = details['name'] # Expect 'name' to be the instrument_name

        # Find the hedger's data in the main portfolio DataFrame at the current timestamp
        # self.df_portfolio_options should be pre-filtered for relevant expiries if necessary,
        # or contain all options data passed to the class.
        hedger_data_at_ts = self.df_portfolio_options[
            (self.df_portfolio_options['instrument_name'] == hedger_name) &
            (self.df_portfolio_options['date_time'] == timestamp)
        ]

        if hedger_data_at_ts.empty:
            logging.warning(f"Hedger {hedger_name} data not found at timestamp {timestamp}")
            return np.nan, np.nan, np.nan

        hedger_row_series = hedger_data_at_ts.iloc[0].copy() # Ensure it's a Series

        current_hedger_iv = hedger_row_series.get('iv_close', np.nan)
        current_hedger_mark_price = hedger_row_series.get('mark_price_close', np.nan)

        if pd.isna(current_hedger_iv) or current_hedger_iv <= 0:
            logging.warning(f"Hedger {hedger_name} IV invalid at {timestamp}: {current_hedger_iv}")
            # Return NaNs for greeks, but price might still be valid if IV is the only issue
            return np.nan, np.nan, current_hedger_mark_price

        # Pass the full row (which includes k, option_type, expiry_datetime_col, iv_close)
        hedger_delta = compute_delta(hedger_row_series, spot_price, timestamp, self.risk_free_rate)
        hedger_gamma = compute_gamma(hedger_row_series, spot_price, timestamp, self.risk_free_rate)
        
        MIN_GAMMA_FOR_GREEK_VALIDITY = 1e-7 # A small threshold to consider gamma valid
        if pd.isna(hedger_delta) or pd.isna(hedger_gamma) or abs(hedger_gamma) < MIN_GAMMA_FOR_GREEK_VALIDITY:
            logging.warning(f"Hedger {hedger_name} greeks invalid or gamma too small at {timestamp}. D={hedger_delta}, G={hedger_gamma}")
            return np.nan, np.nan, current_hedger_mark_price

        return hedger_delta, hedger_gamma, current_hedger_mark_price

    def _solve_delta_gamma_hedge_system(self, portfolio_value, portfolio_delta, portfolio_gamma, spot_price, hedger_delta, hedger_gamma, hedger_price):
        if pd.isna(portfolio_delta) or pd.isna(portfolio_gamma) or pd.isna(spot_price) or pd.isna(hedger_delta) or pd.isna(hedger_gamma): return np.nan, np.nan, np.nan
        if pd.isna(hedger_price) or pd.isna(portfolio_value):
            A = np.array([[1.0, hedger_delta], [0.0, hedger_gamma]]); b = np.array([-portfolio_delta, -portfolio_gamma])
            if abs(np.linalg.det(A)) < 1e-9: return np.nan, np.nan, np.nan
            try: x = np.linalg.solve(A, b); return np.nan, x[0], x[1]
            except np.linalg.LinAlgError: return np.nan, np.nan, np.nan
        else:
            A = np.array([[-1.0, spot_price, hedger_price], [0.0, 1.0, hedger_delta], [0.0, 0.0, hedger_gamma]]); b = np.array([-portfolio_value, -portfolio_delta, -portfolio_gamma])
            if abs(np.linalg.det(A)) < 1e-9: return np.nan, np.nan, np.nan
            try: x = np.linalg.solve(A, b); return x[0], x[1], x[2]
            except np.linalg.LinAlgError: return np.nan, np.nan, np.nan

    def run_loop(self, days=5):
        try: # Top-level try for the whole method
            # --- 1. Initial Data Validation and Range Setup ---
            if self.df_portfolio_options.empty or self.spot_df.empty:
                logging.info("MatrixDeltaGammaHedgeSimple.run_loop: df_portfolio_options or spot_df is empty at start. Exiting.")
                return pd.DataFrame(), pd.DataFrame()

            latest_hist_ts = self.df_portfolio_options['date_time'].max()
            latest_spot_ts = self.spot_df['date_time'].max()
            if pd.isna(latest_hist_ts) or pd.isna(latest_spot_ts):
                logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: latest_hist_ts ({latest_hist_ts}) or latest_spot_ts ({latest_spot_ts}) is NaN. Exiting.")
                return pd.DataFrame(), pd.DataFrame()

            latest_timestamp = min(latest_hist_ts, latest_spot_ts)
            
            # Ensure min_data_ts is valid before using in max()
            min_portfolio_ts = self.df_portfolio_options['date_time'].min()
            min_spot_ts = self.spot_df['date_time'].min()
            if pd.isna(min_portfolio_ts) or pd.isna(min_spot_ts):
                logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: min_portfolio_ts ({min_portfolio_ts}) or min_spot_ts ({min_spot_ts}) is NaN. Exiting.")
                return pd.DataFrame(), pd.DataFrame()
            min_data_ts = max(min_portfolio_ts, min_spot_ts)


            potential_start_timestamp = latest_timestamp - pd.Timedelta(days=days)
            start_timestamp = max(potential_start_timestamp, min_data_ts)

            if start_timestamp >= latest_timestamp:
                logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: start_timestamp ({start_timestamp}) >= latest_timestamp ({latest_timestamp}). No sim range. Exiting.")
                return pd.DataFrame(), pd.DataFrame()

            logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: Simulation period from {start_timestamp} to {latest_timestamp}.")

            # --- 2. Filter Data for Simulation Period ---
            sim_options_df = self.df_portfolio_options[
                (self.df_portfolio_options['date_time'] >= start_timestamp) &
                (self.df_portfolio_options['date_time'] <= latest_timestamp)
            ].copy()
            sim_spot_df = self.spot_df[
                (self.spot_df['date_time'] >= start_timestamp) &
                (self.spot_df['date_time'] <= latest_timestamp)
            ].copy()

            if sim_options_df.empty or sim_spot_df.empty:
                logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: sim_options_df (empty: {sim_options_df.empty}) or sim_spot_df (empty: {sim_spot_df.empty}) is empty after filtering for sim range. Exiting.")
                return pd.DataFrame(), pd.DataFrame()

            # --- 3. Determine Driving Timestamps for Simulation ---
            # Use unique timestamps from the options data as the primary loop driver
            loop_driving_timestamps_array = sim_options_df['date_time'].unique()
            if len(loop_driving_timestamps_array) == 0:
                 logging.info("MatrixDeltaGammaHedgeSimple.run_loop: No unique timestamps in sim_options_df. Exiting.")
                 return pd.DataFrame(), pd.DataFrame()
            loop_driving_timestamps = sorted(loop_driving_timestamps_array)


            loop_timestamps_df = pd.DataFrame({'date_time': loop_driving_timestamps})

            # Merge spot data onto these driving timestamps
            spot_for_sim = pd.merge_asof(
                left=loop_timestamps_df.sort_values('date_time'), # Ensure left is sorted
                right=sim_spot_df[['date_time', 'close']].sort_values('date_time'), # Ensure right is sorted
                on='date_time',
                direction='backward', # Use last known spot if exact match not found
                tolerance=pd.Timedelta('10min') # Allow some gap
            )
            # Forward fill then backward fill to handle NaNs from merge_asof or initial NaNs in spot data
            spot_for_sim['close'] = spot_for_sim['close'].ffill().bfill()
            spot_for_sim = spot_for_sim.dropna(subset=['date_time', 'close']) # Remove rows where spot is still NaN

            if spot_for_sim.empty:
                logging.info("MatrixDeltaGammaHedgeSimple.run_loop: spot_for_sim is empty after merge_asof and NaN handling. Exiting.")
                return pd.DataFrame(), pd.DataFrame()

            # --- 4. Refine Final Simulation Timestamps ---
            # Ensure final_sim_timestamps are only those where we have valid spot data
            final_sim_timestamps_defined = False
            final_sim_timestamps = []
            try:
                if 'date_time' not in spot_for_sim.columns:
                    logging.error("MatrixDeltaGammaHedgeSimple.run_loop: 'date_time' column missing in spot_for_sim. Exiting.")
                    return pd.DataFrame(), pd.DataFrame()
                
                unique_dates_array_final = spot_for_sim['date_time'].unique()
                if len(unique_dates_array_final) > 0:
                    final_sim_timestamps = sorted(unique_dates_array_final)
                    final_sim_timestamps_defined = True
                logging.debug(f"MatrixDeltaGammaHedgeSimple.run_loop: Final unique timestamps for sim: {len(final_sim_timestamps)}")

            except Exception as e_final_ts:
                logging.error(f"MatrixDeltaGammaHedgeSimple.run_loop: Error processing final sim timestamps: {e_final_ts}", exc_info=True)
                return pd.DataFrame(), pd.DataFrame()

            if not final_sim_timestamps_defined or not final_sim_timestamps:
                logging.info("MatrixDeltaGammaHedgeSimple.run_loop: No valid final simulation timestamps after processing spot_for_sim. Exiting.")
                return pd.DataFrame(), pd.DataFrame()
            
            logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: Proceeding with {len(final_sim_timestamps)} simulation timestamps.")

            # --- 5. Initialize Hedging State and Logs ---
            self.portfolio_state_log = []
            self.hedge_actions_log = []
            self.current_underlying_hedge_qty = 0.0
            self.current_gamma_option_hedge_qty = 0.0
            trade_tolerance = 1e-6  # Minimum trade size to register
            MIN_EFFECTIVE_HEDGER_GAMMA = 1e-5 # Minimum gamma for hedger to be considered effective

            # --- 6. Main Simulation Loop ---
            for ts in final_sim_timestamps:
                try:
                    # Get spot price for the current timestamp
                    spot_price_at_ts_series = spot_for_sim.loc[spot_for_sim['date_time'] == ts, 'close']
                    if spot_price_at_ts_series.empty:
                        logging.warning(f"Timestamp {ts}: Spot price not found in spot_for_sim. Skipping step.")
                        continue # Should not happen if final_sim_timestamps is derived from spot_for_sim
                    spot_price_at_ts = spot_price_at_ts_series.iloc[0]
                    
                    if pd.isna(spot_price_at_ts) or spot_price_at_ts <= 0:
                        logging.warning(f"Timestamp {ts}: Invalid spot price ({spot_price_at_ts}). Skipping step.")
                        continue

                    # Get portfolio greeks at current timestamp and spot
                    port_val, port_delta, port_gamma = self._get_portfolio_greeks(ts, spot_price_at_ts)
                    if pd.isna(port_delta) or pd.isna(port_gamma):
                        logging.warning(f"Timestamp {ts}: Portfolio greeks NaN (Delta={port_delta}, Gamma={port_gamma}). Skipping step.")
                        # Log a NaN state for this timestamp to maintain series
                        self.portfolio_state_log.append({'timestamp': ts, 'spot_price': spot_price_at_ts, 'portfolio_delta': np.nan, 'portfolio_gamma': np.nan, 'current_n_underlying': self.current_underlying_hedge_qty, 'current_n_gamma_opt': self.current_gamma_option_hedge_qty, 'net_delta_final':np.nan, 'net_gamma_final':np.nan})
                        continue

                    # Get gamma hedger greeks and price
                    hedger_D_current, hedger_G_current, hedger_P_current = np.nan, np.nan, np.nan
                    can_gamma_hedge_this_step = False

                    if self.gamma_hedge_instrument_details:
                        hedger_D_temp, hedger_G_temp, hedger_P_temp = self._get_gamma_hedger_greeks_and_price(ts, spot_price_at_ts)
                        if pd.notna(hedger_G_temp) and abs(hedger_G_temp) >= MIN_EFFECTIVE_HEDGER_GAMMA and pd.notna(hedger_D_temp):
                            hedger_D_current, hedger_G_current, hedger_P_current = hedger_D_temp, hedger_G_temp, hedger_P_temp
                            can_gamma_hedge_this_step = True
                        else:
                            logging.debug(f"Timestamp {ts}: Gamma hedger unsuitable. G={hedger_G_temp}, D={hedger_D_temp}. Gamma hedge option trade will not be made this step.")
                    
                    # Step 6.1: Calculate and apply gamma option hedge if feasible
                    actual_trade_size_gamma_opt = 0.0
                    if can_gamma_hedge_this_step:
                        target_total_gamma_opt_qty = -port_gamma / hedger_G_current # Target total units of gamma option
                        trade_size_gamma_opt_ideal = target_total_gamma_opt_qty - self.current_gamma_option_hedge_qty
                        
                        if abs(trade_size_gamma_opt_ideal) > trade_tolerance:
                            actual_trade_size_gamma_opt = trade_size_gamma_opt_ideal
                            self.hedge_actions_log.append({
                                'timestamp': ts, 'instrument': self.gamma_hedge_instrument_details['name'],
                                'action': 'buy' if actual_trade_size_gamma_opt > 0 else 'sell',
                                'size': abs(actual_trade_size_gamma_opt),
                                'price': hedger_P_current if pd.notna(hedger_P_current) else np.nan,
                                'type': 'gamma_option'
                            })
                            self.current_gamma_option_hedge_qty += actual_trade_size_gamma_opt
                    
                    # Step 6.2: Calculate delta to be hedged by underlying
                    # This is: (Portfolio Delta) + (Delta from ALL existing gamma options)
                    delta_to_be_hedged_by_underlying = port_delta
                    if self.current_gamma_option_hedge_qty != 0: # If we hold any gamma options
                        if can_gamma_hedge_this_step and pd.notna(hedger_D_current): # And current hedger delta is valid
                            delta_to_be_hedged_by_underlying += self.current_gamma_option_hedge_qty * hedger_D_current
                        elif self.gamma_hedge_instrument_details: # We hold gamma options, but current hedger delta is bad
                             # This case implies the hedger instrument details are problematic at this step.
                             # For simplicity, if hedger_D_current is NaN, we can't accurately assess its delta impact.
                             # A more complex model might try to use the last known good delta or a proxy.
                             # Here, we proceed, effectively assuming its delta impact is zero if hedger_D_current is NaN for this step.
                            logging.warning(f"Timestamp {ts}: Delta of existing gamma option position ({self.current_gamma_option_hedge_qty} units of {self.gamma_hedge_instrument_details['name']}) is uncertain as hedger_D_current is {hedger_D_current}. Underlying hedge might be imprecise.")

                    # Step 6.3: Calculate and apply underlying hedge
                    target_total_underlying_qty = -delta_to_be_hedged_by_underlying # Target total units of underlying
                    trade_size_underlying = target_total_underlying_qty - self.current_underlying_hedge_qty
                    
                    if abs(trade_size_underlying) > trade_tolerance:
                        self.hedge_actions_log.append({
                            'timestamp': ts, 'instrument': self.symbol + '-PERP', # Assuming PERP for underlying hedge
                            'action': 'buy' if trade_size_underlying > 0 else 'sell',
                            'size': abs(trade_size_underlying),
                            'price': spot_price_at_ts,
                            'type': 'delta_underlying'
                        })
                        self.current_underlying_hedge_qty += trade_size_underlying

                    # Step 6.4: Log portfolio state *after* hedges for this timestamp
                    net_delta_final = port_delta + self.current_underlying_hedge_qty # Delta from underlying
                    net_gamma_final = port_gamma                                    # Gamma from portfolio

                    if self.current_gamma_option_hedge_qty != 0: # If we hold gamma options
                        if can_gamma_hedge_this_step and pd.notna(hedger_D_current) and pd.notna(hedger_G_current):
                            net_delta_final += self.current_gamma_option_hedge_qty * hedger_D_current
                            net_gamma_final += self.current_gamma_option_hedge_qty * hedger_G_current
                        # If hedger greeks are NaN at this step but we hold positions, net_delta/gamma might not reflect their true contribution
                    
                    self.portfolio_state_log.append({
                        'timestamp': ts, 'spot_price': spot_price_at_ts,
                        'portfolio_value': port_val, # Original portfolio value before hedges for this step
                        'portfolio_delta': port_delta, 'portfolio_gamma': port_gamma, # Original portfolio greeks
                        'target_B': np.nan, # Not directly calculated/used in this stepwise approach
                        'target_n_underlying': self.current_underlying_hedge_qty, # Total underlying after this step's trade
                        'target_n_gamma_opt': self.current_gamma_option_hedge_qty, # Total gamma option after this step's trade
                        'current_n_underlying': self.current_underlying_hedge_qty,
                        'current_n_gamma_opt': self.current_gamma_option_hedge_qty,
                        'hedger_delta_at_ts': hedger_D_current if can_gamma_hedge_this_step else np.nan,
                        'hedger_gamma_at_ts': hedger_G_current if can_gamma_hedge_this_step else np.nan,
                        'hedger_price_at_ts': hedger_P_current if can_gamma_hedge_this_step else np.nan,
                        'net_delta_final': net_delta_final, # Net delta of (portfolio + all hedges)
                        'net_gamma_final': net_gamma_final  # Net gamma of (portfolio + all hedges)
                    })

                except Exception as e_loop_iter:
                    logging.error(f"Error in simulation loop iteration at timestamp {ts}: {e_loop_iter}", exc_info=True)
                    self.portfolio_state_log.append({
                        'timestamp': ts, 'spot_price': np.nan, 'portfolio_value':np.nan,
                        'portfolio_delta':np.nan, 'portfolio_gamma':np.nan, 'target_B':np.nan,
                        'target_n_underlying':self.current_underlying_hedge_qty,
                        'target_n_gamma_opt':self.current_gamma_option_hedge_qty,
                        'current_n_underlying':self.current_underlying_hedge_qty,
                        'current_n_gamma_opt':self.current_gamma_option_hedge_qty,
                        'hedger_delta_at_ts':np.nan, 'hedger_gamma_at_ts':np.nan,
                        'hedger_price_at_ts':np.nan, 'net_delta_final':np.nan, 'net_gamma_final':np.nan
                    })
            # --- End of Main Simulation Loop ---

            logging.info(f"MatrixDeltaGammaHedgeSimple.run_loop: Simulation finished. Processed {len(final_sim_timestamps)} timestamps.")
            return pd.DataFrame(self.portfolio_state_log), pd.DataFrame(self.hedge_actions_log)

        except Exception as e_outer_run_loop:
            logging.error(f"CRITICAL UNHANDLED ERROR in MatrixDeltaGammaHedgeSimple.run_loop: {e_outer_run_loop}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame() # Ensure a two-DataFrame return
        # ---- End of robust creation ----
def plot_mm_delta_gamma_hedge(portfolio_state_df, hedge_actions_df, symbol):
    st.subheader(f"MM Delta-Gamma Hedging Simulation Visuals ({symbol})")
    if portfolio_state_df.empty: return
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Net Portfolio Greeks (Delta & Gamma)", f"Underlying Hedge Position ({symbol}) & Spot Price", "Gamma Option Hedge Position"), specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": False}]])
    if 'net_delta_final' in portfolio_state_df.columns: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['net_delta_final'], mode='lines', name='Net Delta', line=dict(color='cyan')), secondary_y=False, row=1, col=1)
    if 'net_gamma_final' in portfolio_state_df.columns: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['net_gamma_final'], mode='lines', name='Net Gamma', line=dict(color='magenta')), secondary_y=True, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=1, secondary_y=False)
    if 'current_n_underlying' in portfolio_state_df.columns: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['current_n_underlying'], mode='lines', name=f'Underlying Hedge ({symbol})', line=dict(color='lightgreen')), secondary_y=False, row=2, col=1)
    if 'spot_price' in portfolio_state_df.columns: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['spot_price'], mode='lines', name='Spot Price', line=dict(color='grey', dash='dash')), secondary_y=True, row=2, col=1)
    if 'current_n_gamma_opt' in portfolio_state_df.columns: fig.add_trace(go.Scatter(x=portfolio_state_df['timestamp'], y=portfolio_state_df['current_n_gamma_opt'], mode='lines', name='Gamma Option Hedge Qty', line=dict(color='orange')), row=3, col=1)
    if not hedge_actions_df.empty and 'type' in hedge_actions_df.columns:
        underlying_trades = hedge_actions_df[hedge_actions_df['type'] == 'delta_underlying']; gamma_option_trades = hedge_actions_df[hedge_actions_df['type'] == 'gamma_option']
        for _, trade in underlying_trades.iterrows():
            y_val = portfolio_state_df[portfolio_state_df['timestamp'] == trade['timestamp']]['current_n_underlying'].iloc[0] if not portfolio_state_df[portfolio_state_df['timestamp'] == trade['timestamp']].empty else np.nan
            fig.add_trace(go.Scatter(x=[trade['timestamp']], y=[y_val], mode='markers', marker=dict(symbol='triangle-up' if trade['action']=='buy' else 'triangle-down', size=8, color='lime' if trade['action']=='buy' else 'red'), name=f"{trade['action']} Underlying", showlegend=False), row=2, col=1, secondary_y=False)
        for _, trade in gamma_option_trades.iterrows():
            y_val_gamma = portfolio_state_df[portfolio_state_df['timestamp'] == trade['timestamp']]['current_n_gamma_opt'].iloc[0] if not portfolio_state_df[portfolio_state_df['timestamp'] == trade['timestamp']].empty else np.nan
            fig.add_trace(go.Scatter(x=[trade['timestamp']], y=[y_val_gamma], mode='markers', marker=dict(symbol='circle', size=7, color='yellow' if trade['action']=='buy' else 'purple'), name=f"{trade['action']} Gamma Option", showlegend=False), row=3, col=1)
    fig.update_layout(height=900, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Net Delta", secondary_y=False, row=1, col=1); fig.update_yaxes(title_text="Net Gamma", secondary_y=True, row=1, col=1, tickformat=".2e"); fig.update_yaxes(title_text="Underlying Qty", secondary_y=False, row=2, col=1); fig.update_yaxes(title_text="Spot Price", secondary_y=True, row=2, col=1, showgrid=False); fig.update_yaxes(title_text="Gamma Option Qty", row=3, col=1); fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
def compute_and_plot_itm_gex_ratio(dft, df_krak_5m, spot_price_latest, selected_expiry_obj):
    expiry_label = "N/A"; coin = "N/A"
    if isinstance(selected_expiry_obj, dt.datetime):
         try: expiry_label = selected_expiry_obj.strftime('%d%b%y')
         except ValueError: pass
    if not dft.empty and 'instrument_name' in dft.columns:
        try: coin = dft['instrument_name'].iloc[0].split('-')[0]
        except Exception: pass
    st.subheader(f"ITM Put/Call GEX Ratio (Expiry: {expiry_label})")
    required_dft_cols = ['date_time', 'k', 'option_type', 'iv_close', 'open_interest', 'instrument_name']; required_spot_cols = ['date_time', 'close']
    if dft.empty or df_krak_5m.empty or not all(c in dft.columns for c in required_dft_cols) or not all(c in df_krak_5m.columns for c in required_spot_cols): return
    if not pd.api.types.is_datetime64_any_dtype(dft['date_time']) or dft['date_time'].dt.tz != dt.timezone.utc: return
    if not pd.api.types.is_datetime64_any_dtype(df_krak_5m['date_time']) or df_krak_5m['date_time'].dt.tz != dt.timezone.utc: return
    try:
        dft_local = dft.copy(); spot_local = df_krak_5m.copy()
        dft_local = dft_local.sort_values('date_time'); spot_local = spot_local.sort_values('date_time')
        df_merged = pd.merge_asof(dft_local, spot_local[['date_time', 'close']].rename(columns={'close': 'spot_price'}), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min'))
        essential_cols = ['date_time', 'spot_price', 'k', 'option_type', 'iv_close', 'open_interest', 'instrument_name']
        df_merged = df_merged.dropna(subset=essential_cols)
        if df_merged.empty: return
        for col_to_convert in ['spot_price', 'k', 'iv_close', 'open_interest']: df_merged[col_to_convert] = pd.to_numeric(df_merged[col_to_convert], errors='coerce')
        df_merged = df_merged.dropna(subset=['spot_price', 'k', 'iv_close', 'open_interest'])
        if df_merged.empty: return
        with st.spinner(f"Calculating timestamped Gamma & GEX (Expiry: {expiry_label})..."):
            df_merged['gamma'] = df_merged.apply(lambda row: compute_gamma(row, row['spot_price'], row['date_time']), axis=1)
            df_merged['gex'] = df_merged.apply(lambda row: compute_gex(row, row['spot_price'], row['open_interest']), axis=1)
        df_merged = df_merged.dropna(subset=['gex'])
        if df_merged.empty: return
        df_merged['is_itm_call'] = (df_merged['option_type'] == 'C') & (df_merged['k'] < df_merged['spot_price']); df_merged['is_itm_put'] = (df_merged['option_type'] == 'P') & (df_merged['k'] > df_merged['spot_price'])
        def aggregate_gex(group): return pd.Series({'total_itm_call_gex': group.loc[group['is_itm_call'], 'gex'].sum(skipna=True), 'total_itm_put_gex': group.loc[group['is_itm_put'], 'gex'].sum(skipna=True)})
        df_agg = df_merged.groupby('date_time').apply(aggregate_gex, include_groups=False).reset_index()
        epsilon = 1e-9; df_agg['itm_gex_ratio'] = df_agg['total_itm_put_gex'] / (df_agg['total_itm_call_gex'] + epsilon)
        df_agg['itm_gex_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True); df_plot = df_agg.dropna(subset=['itm_gex_ratio'])
        min_timestamps_for_plot = 10
        if df_plot.empty or len(df_plot) < min_timestamps_for_plot: return
        df_plot = pd.merge_asof(df_plot.sort_values('date_time'), spot_local[['date_time', 'close']].sort_values('date_time'), on='date_time', direction='nearest', tolerance=pd.Timedelta('5min')).dropna(subset=['close'])
        if df_plot.empty: return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['itm_gex_ratio'], name='ITM Put/Call GEX Ratio', mode='lines', line=dict(color='mediumseagreen', width=1.5), fill='tozeroy', fillcolor='rgba(60, 179, 113, 0.3)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['close'], name=f'{coin} Spot Price', mode='lines', line=dict(color='cornflowerblue', width=2)), secondary_y=True)
        title_coin = coin if coin != "N/A" else "Crypto"; title_text = f"{title_coin} Intraday - ${spot_price_latest:,.2f}, ITM PUT/CALL GEX Ratio"
        fig.update_layout(title=title_text, height=500, xaxis_title="Date (UTC)", yaxis_title="Ratio", yaxis2_title="Price", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), hovermode='x unified', plot_bgcolor='rgba(17, 17, 17, 1)', paper_bgcolor='rgba(17, 17, 17, 1)', font=dict(color='lightgrey'), xaxis=dict(gridcolor='rgba(68, 68, 68, 0.5)'), yaxis=dict(gridcolor='rgba(68, 68, 68, 0.5)'), yaxis2=dict(showgrid=False))
        fig.update_xaxes(tickformat="%m/%d\n%H:%M", nticks=10); fig.update_yaxes(rangemode='tozero', secondary_y=False, zeroline=True, zerolinewidth=1, zerolinecolor='grey')
        if not df_plot.empty and 'close' in df_plot.columns:
            min_price_plot = df_plot['close'].min(); max_price_plot = df_plot['close'].max()
            if pd.notna(min_price_plot) and pd.notna(max_price_plot): fig.update_yaxes(range=[min_price_plot * 0.99, max_price_plot * 1.01], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        if not df_plot.empty:
            latest_ratio = df_plot['itm_gex_ratio'].iloc[-1]
            st.metric("Latest ITM Put/Call GEX Ratio", f"{latest_ratio:.2f}" if pd.notna(latest_ratio) else "N/A")
        # Call the styled plot if needed
        plot_gex_dashboard_image_style(df_plot_data=df_plot, spot_price_latest=spot_price_latest, coin_symbol=coin, expiry_label_for_title=expiry_label)
    except Exception: pass


def plot_gex_dashboard_image_style(df_plot_data, spot_price_latest, coin_symbol, expiry_label_for_title):
    st.subheader(f"Alternative View: {coin_symbol} Intraday ITM PUT/CALL GEX Ratio")
    if df_plot_data.empty or not all(col in df_plot_data.columns for col in ['date_time', 'itm_gex_ratio', 'close']): return
    df_plot = df_plot_data.copy(); df_plot['date_time'] = pd.to_datetime(df_plot['date_time']); df_plot = df_plot.sort_values('date_time').dropna(subset=['itm_gex_ratio', 'close'])
    if df_plot.empty or len(df_plot) < 2: return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['itm_gex_ratio'], name='Ratio (ITM P/C GEX)', mode='lines', line=dict(color='dodgerblue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot['date_time'], y=df_plot['close'], name=f'{coin_symbol} Price', mode='lines', line=dict(color='mediumseagreen', width=1.5), fill='tozeroy', fillcolor='rgba(60, 179, 113, 0.3)'), secondary_y=True)
    year_for_circles = 2025; circles_definitions = [{"x0_str": f"{year_for_circles}-03-29 12:00:00", "x1_str": f"{year_for_circles}-03-30 10:00:00"}, {"x0_str": f"{year_for_circles}-03-30 20:00:00", "x1_str": f"{year_for_circles}-03-31 10:00:00"}, {"x0_str": f"{year_for_circles}-04-01 00:00:00", "x1_str": f"{year_for_circles}-04-01 09:00:00"}]
    shapes = []
    for circle_def in circles_definitions:
        try:
            x0_dt = pd.to_datetime(circle_def["x0_str"]).tz_localize('UTC'); x1_dt = pd.to_datetime(circle_def["x1_str"]).tz_localize('UTC')
            min_data_time = df_plot['date_time'].min(); max_data_time = df_plot['date_time'].max()
            if not (x1_dt < min_data_time or x0_dt > max_data_time): shapes.append(dict(type="circle", xref="x", yref="paper", x0=x0_dt, y0=0.05, x1=x1_dt, y1=0.95, line_color="red", line_width=2, opacity=0.7, layer="below"))
        except Exception: pass
    title_text = f"{coin_symbol} Intraday - ${spot_price_latest:,.2f}, ITM PUT/CALL GEX Ratio"
    fig.update_layout(title=title_text, height=600, plot_bgcolor='rgba(17, 17, 17, 1)', paper_bgcolor='rgba(17, 17, 17, 1)', font=dict(color='lightgrey'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(30,30,30,0.7)', bordercolor='grey', borderwidth=1), xaxis=dict(title="Date (UTC)", gridcolor='rgba(68, 68, 68, 0.5)', linecolor='grey', showline=True, zeroline=False, tickformat="%m/%d\n%H:%M"), yaxis=dict(title="Ratio", side='left', gridcolor='rgba(68, 68, 68, 0.5)', linecolor='grey', showline=True, zeroline=False, range=[0, 20]), yaxis2=dict(title="Price", side='right', overlaying='y', showgrid=False, linecolor='grey', showline=True, zeroline=False, tickprefix="$", tickformat=",.0fK"), hovermode='x unified', shapes=shapes, annotations=[dict(text="CheapGamma.com", align='right', showarrow=False, xref='paper', yref='paper', x=0.99, y=1.06, font=dict(color='purple', size=12))])
    try: st.plotly_chart(fig, use_container_width=True)
    except Exception: pass    
# --- Main Function Definition ---
def main():
    st.set_page_config(layout="wide", page_title="Delta Hedging & MM Focus")
    login()

    if 'selected_coin' not in st.session_state: st.session_state.selected_coin = "BTC"
    if 'snapshot_time' not in st.session_state: st.session_state.snapshot_time = dt.datetime.now(dt.timezone.utc)
    if 'risk_free_rate_input' not in st.session_state: st.session_state.risk_free_rate_input = 0.01

    st.title(f"{st.session_state.selected_coin} Options: Delta Hedging & MM Perspective")

    if st.sidebar.button("Logout"):
        keys_to_clear = [k for k in st.session_state.keys() if k != 'logged_in']; [st.session_state.pop(key) for key in keys_to_clear]; st.session_state.logged_in = False; st.rerun()

    dft = pd.DataFrame(); dft_latest = pd.DataFrame(); ticker_data = {}
    df_krak_5m = pd.DataFrame(); df_krak_daily = pd.DataFrame()
    all_instruments_list = []; expiry_options = []; all_instr_selected_expiry = []
    selected_expiry = None; e_str = ""; spot_price = np.nan
    all_calls_expiry = []; all_puts_expiry = []
    selected_call_instr_ideal = None; selected_put_instr_ideal = None

    st.sidebar.header("Configuration")
    coin_options = ["BTC", "ETH"]
    if st.session_state.selected_coin not in coin_options: st.session_state.selected_coin = "BTC"
    current_coin_index = coin_options.index(st.session_state.selected_coin)
    selected_coin_from_widget = st.sidebar.selectbox("Select Cryptocurrency", coin_options, index=current_coin_index, key='coin_selector_focused_v3')
    if selected_coin_from_widget != st.session_state.selected_coin: st.session_state.selected_coin = selected_coin_from_widget; st.rerun()
    coin = st.session_state.selected_coin
    st.session_state.risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate", value=st.session_state.get('risk_free_rate_input', 0.01), min_value=0.0, max_value=0.2, step=0.001, format="%.3f", key="rf_rate_focused_v3")
    risk_free_rate = st.session_state.risk_free_rate_input

    with st.spinner("Fetching Thalex instruments..."): all_instruments_list = fetch_instruments()
    if not all_instruments_list: st.error("Failed to fetch Thalex instrument list."); st.stop()
    now_utc = dt.datetime.now(dt.timezone.utc)
    with st.spinner("Determining expiries..."): expiry_options = get_valid_expiration_options(now_utc)
    if not expiry_options: st.error(f"No valid future expiries for {coin}."); st.stop()
    default_expiry_idx = 0
    if expiry_options:
        for i, exp_dt in enumerate(expiry_options):
            if get_option_instruments(all_instruments_list, "C", exp_dt.strftime("%d%b%y").upper(), coin): default_expiry_idx = i; break
    selected_expiry = st.sidebar.selectbox("Select Expiry Date", options=expiry_options, format_func=lambda dt_obj: dt_obj.strftime("%d %b %Y"), index=default_expiry_idx, key=f"expiry_selector_focused_v3_{coin}")
    if selected_expiry: e_str = selected_expiry.strftime("%d%b%y").upper()
    else: st.error("Please select an expiry date."); st.stop()
    all_calls_expiry = get_option_instruments(all_instruments_list, "C", e_str, coin)
    all_puts_expiry = get_option_instruments(all_instruments_list, "P", e_str, coin)
    all_instr_selected_expiry = sorted(all_calls_expiry + all_puts_expiry)

    with st.spinner(f"Fetching Kraken {coin} spot data..."):
        df_krak_5m = fetch_kraken_data(coin=coin, days=7)
        df_krak_daily = fetch_kraken_data_daily(days=365, coin=coin)
    if df_krak_5m.empty or df_krak_daily.empty: st.error(f"Failed to fetch Kraken {coin} data."); st.stop()
    spot_price = df_krak_5m["close"].iloc[-1] if not df_krak_5m.empty else np.nan
    if pd.isna(spot_price): st.error("Could not determine latest spot price."); st.stop()
    
    st.header(f"MM & Delta Hedge Analysis: {coin} | Expiry: {selected_expiry.strftime('%d %b %Y')} | Spot: ${spot_price:,.2f}")

    if not all_instr_selected_expiry: st.error(f"No {coin} options for selected expiry {e_str}."); st.stop()
    with st.spinner(f"Fetching historical options data for expiry {e_str}..."): dft = fetch_data(tuple(all_instr_selected_expiry))
    required_cols_dft = ['date_time', 'instrument_name', 'k', 'option_type', 'mark_price_close', 'iv_close', 'expiry_datetime_col']
    if dft.empty or not all(col in dft.columns for col in required_cols_dft): st.error(f"Failed to process options for {e_str}."); st.stop()
    if 'iv_close' in dft.columns and dft['iv_close'].notna().any():
        dft['iv_close'] = pd.to_numeric(dft['iv_close'], errors='coerce')
        if dft['iv_close'].abs().max() > 1.5: dft['iv_close'] /= 100.0
    with st.spinner(f"Fetching latest ticker data..."): ticker_data = {instr: fetch_ticker(instr) for instr in all_instr_selected_expiry}
    dft['open_interest'] = dft['instrument_name'].map(lambda x: ticker_data.get(x, {}).get('open_interest', 0.0)).astype('float32')

    df_merged_for_greeks = pd.DataFrame()
    if not df_krak_5m.empty:
        df_merged_for_greeks = merge_spot_to_options(dft, df_krak_5m, selected_expiry)
        if not df_merged_for_greeks.empty:
            with st.spinner("Calculating historical Greeks..."):
                df_merged_for_greeks["delta"] = df_merged_for_greeks.apply(lambda r: compute_delta(r, r['spot_price'], r['date_time'], risk_free_rate), axis=1).astype('float32')
                df_merged_for_greeks["gamma"] = df_merged_for_greeks.apply(lambda r: compute_gamma(r, r['spot_price'], r['date_time'], risk_free_rate), axis=1).astype('float32')
            dft = df_merged_for_greeks
        else: dft["delta"] = np.nan; dft["gamma"] = np.nan
    else: dft["delta"] = np.nan; dft["gamma"] = np.nan

    dft_latest = pd.DataFrame()
    if not dft.empty and 'date_time' in dft.columns:
        try:
            latest_indices = dft.groupby('instrument_name')['date_time'].idxmax(); dft_latest_temp = dft.loc[latest_indices].copy()
            greek_cols_check = ['instrument_name', 'k', 'iv_close', 'option_type', 'expiry_datetime_col']
            if all(c in dft_latest_temp.columns for c in greek_cols_check):
                current_time = st.session_state.snapshot_time
                dft_latest_temp["delta"] = dft_latest_temp.apply(lambda r: compute_delta(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["gamma"] = dft_latest_temp.apply(lambda r: compute_gamma(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["vega"] = dft_latest_temp.apply(lambda r: compute_vega(r, spot_price, current_time), axis=1).astype('float32')
                dft_latest_temp["charm"] = dft_latest_temp.apply(lambda r: compute_charm(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["vanna"] = dft_latest_temp.apply(lambda r: compute_vanna(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                if 'gamma' in dft_latest_temp.columns and 'open_interest' in dft_latest_temp.columns: dft_latest_temp["gex"] = dft_latest_temp.apply(lambda r: compute_gex(r, spot_price, r['open_interest']), axis=1).astype('float32')
                else: dft_latest_temp["gex"] = np.nan
                dft_latest = dft_latest_temp
        except Exception: dft_latest = pd.DataFrame()

    def safe_plot(plot_func, *args, **kwargs):
        try:
            if callable(plot_func): plot_func(*args, **kwargs)
        except Exception as e: st.error(f"Plot error in '{getattr(plot_func, '__name__', 'N/A')}'. Check logs."); logging.error(f"Plot error", exc_info=True)

    # =========================== Key Metrics & Market Memory ==========================
    st.markdown("---"); st.header("Understanding Key Metrics & Market Memory")
    col_metrics, col_memory = st.columns([0.4, 0.6])
    with col_metrics:
        st.subheader("Volatility Snapshot")
        rv_30d = calculate_btc_annualized_volatility_daily(df_krak_daily)
        rv_7d_5m = compute_realized_volatility_5min(df_krak_5m.tail(288*7))
        latest_iv_mean_selected = dft_latest['iv_close'].mean() if not dft_latest.empty and 'iv_close' in dft_latest.columns else np.nan
        st.metric("Realized Vol (30d Daily)", f"{rv_30d:.2%}" if pd.notna(rv_30d) else "N/A")
        st.metric("Realized Vol (7d 5-min)", f"{rv_7d_5m:.2%}" if pd.notna(rv_7d_5m) and rv_7d_5m > 0 else "N/A")
        st.metric("Mean IV (Selected Exp.)", f"{latest_iv_mean_selected:.2%}" if pd.notna(latest_iv_mean_selected) else "N/A")
    with col_memory:
        st.subheader("Hurst Exponent (Classic R/S)")
        if not df_krak_daily.empty and 'close' in df_krak_daily:
            daily_log_returns_hurst = np.log(df_krak_daily['close'] / df_krak_daily['close'].shift(1))
            hurst_val, hurst_data_df = calculate_hurst_lo_modified(daily_log_returns_hurst)
            safe_plot(plot_hurst_exponent, hurst_val, hurst_data_df)
        st.markdown("---"); st.subheader("Autocorrelation (Market Memory)")
        if not df_krak_daily.empty and 'close' in df_krak_daily and 'daily_log_returns_hurst' in locals():
            safe_plot(calculate_and_display_autocorrelation, daily_log_returns_hurst, windows=[7, 15, 30])

    # =========================== Market Maker Positioning ==========================
    st.markdown("---"); st.header(f"Market Maker Positioning (Expiry: {selected_expiry.strftime('%d%b%y')})")
    net_delta_mm, net_gex_mm, net_vega_mm, net_vanna_mm, net_charm_mm = np.nan, np.nan, np.nan, np.nan, np.nan
    if not dft_latest.empty:
        if 'delta' in dft_latest.columns and 'open_interest' in dft_latest.columns: net_delta_mm = -1 * (pd.to_numeric(dft_latest['delta'], errors='coerce').fillna(0) * pd.to_numeric(dft_latest['open_interest'], errors='coerce').fillna(0)).sum()
        if 'gex' in dft_latest.columns: net_gex_mm = -1 * dft_latest['gex'].sum(skipna=True)
        if 'vega' in dft_latest.columns: net_vega_mm = -1 * calculate_net_vega(dft_latest)
        if 'vanna' in dft_latest.columns: net_vanna_mm = -1 * calculate_net_vanna(dft_latest)
        if 'charm' in dft_latest.columns: net_charm_mm = -1 * calculate_net_charm(dft_latest)
    col_mm1, col_mm2, col_mm3, col_mm4, col_mm5 = st.columns(5)
    with col_mm1: st.metric("MM Net Delta", f"{net_delta_mm:.2f}" if pd.notna(net_delta_mm) else "N/A")
    with col_mm2: st.metric("MM Net Gamma (GEX)", f"{net_gex_mm:,.0f}" if pd.notna(net_gex_mm) else "N/A")
    with col_mm3: st.metric("MM Net Vega", f"{net_vega_mm:,.0f}" if pd.notna(net_vega_mm) else "N/A")
    with col_mm4: st.metric("MM Net Vanna", f"{net_vanna_mm:,.0f}" if pd.notna(net_vanna_mm) else "N/A")
    with col_mm5: st.metric("MM Net Charm", f"{net_charm_mm:,.2f}" if pd.notna(net_charm_mm) else "N/A")

# =========================== MM Indicative Delta-Gamma Hedge (Snapshot Analysis) ==========================
    st.markdown("---")
    st.header("MM Indicative Delta-Gamma Hedge Adjustment (Latest Snapshot)")
    if not dft_latest.empty and pd.notna(spot_price) and st.session_state.snapshot_time is not None: # Check snapshot_time directly
        # This function 'display_mm_gamma_adjustment_analysis' is for the single point-in-time analysis
        safe_plot(display_mm_gamma_adjustment_analysis, dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)
    else:
        if dft_latest.empty:
            logging.warning("Skipping MM Indicative D-G Snapshot: dft_latest is empty.")
        if not pd.notna(spot_price):
            logging.warning("Skipping MM Indicative D-G Snapshot: spot_price is NaN.")
        if st.session_state.snapshot_time is None:
            logging.warning("Skipping MM Indicative D-G Snapshot: st.session_state.snapshot_time is None.")

    # =========================== MM Delta-Gamma Hedge Simulation Plot ==========================
    # This section is for the historical simulation and its plot
    st.markdown("---") # Optional: Add a separator if you want to visually distinguish it more
    # The header for the simulation plot might be better inside the 'if show_mm_dg_sim_main:' block
    
    show_mm_dg_sim_main = st.sidebar.checkbox("Show MM D-G Hedge Sim Plot", value=True, key="show_mm_dg_sim_main_plot_v3") # Incremented key for uniqueness

    # Initialize DataFrames to ensure they are defined even if the simulation doesn't run
    mm_dg_portfolio_state_df_sim = pd.DataFrame()
    mm_dg_hedge_actions_df_sim = pd.DataFrame()

    if show_mm_dg_sim_main:
        st.header("MM Delta-Gamma Hedging Simulation (Historical)") # Header for this specific section
        if not dft.empty and not df_krak_5m.empty and selected_expiry and not dft_latest.empty:
            # --- Selecting the Gamma Hedging Instrument for the Simulation ---
            # We use dft_latest to pick the instrument based on the *current* market state,
            # but the simulation will use historical data for this instrument's greeks.
            
            # Prefer ATM or slightly OTM calls from the selected expiry
            gamma_hedger_candidate_df = dft_latest[
                (dft_latest['option_type'] == 'C') &
                (dft_latest['k'] >= spot_price) # ATM or OTM calls
            ].sort_values('k') # Sort by strike, ascending (closest to spot_price first among these)

            if gamma_hedger_candidate_df.empty: # Fallback 1: If no ATM/OTM calls, try any call
                all_calls_latest = dft_latest[dft_latest['option_type'] == 'C']
                if not all_calls_latest.empty:
                    # Find the call closest to the spot price (could be ITM)
                    gamma_hedger_candidate_df = all_calls_latest.loc[[
                        abs(all_calls_latest['k'] - spot_price).idxmin()
                    ]]
                    logging.info("MM D-G Sim: No ATM/OTM calls found. Using closest ITM call as fallback for gamma hedger.")
                else:
                    logging.warning("MM D-G Sim: No call options found in dft_latest for the selected expiry.")
            
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
                    # Instantiate the simulation class
                    mm_dg_sim_instance = MatrixDeltaGammaHedgeSimple(
                        df_portfolio_options=dft,       # Full historical data for the portfolio
                        spot_df=df_krak_5m,             # Historical spot data
                        symbol=coin,
                        risk_free_rate=risk_free_rate,
                        gamma_hedge_instrument_details=gamma_hedger_details_sim
                    )
                    logging.info("MM D-G Sim: MatrixDeltaGammaHedgeSimple instance created.")

                    # Run the simulation loop
                    with st.spinner("Running MM Delta-Gamma Hedge Simulation..."):
                        # Ensure 'days' parameter is appropriate for your data range in 'dft' and 'df_krak_5m'
                        sim_days = 5 # Or make this configurable
                        mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim = mm_dg_sim_instance.run_loop(days=sim_days)
                    
                    logging.info(f"MM D-G Sim: run_loop completed. Portfolio states: {len(mm_dg_portfolio_state_df_sim)}, Hedge actions: {len(mm_dg_hedge_actions_df_sim)}")

                    # Plot results if simulation produced data
                    if not mm_dg_portfolio_state_df_sim.empty:
                        safe_plot(plot_mm_delta_gamma_hedge, mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim, coin)
                    else:
                        st.info("MM D-G Sim: Simulation ran but produced no portfolio state data to plot.")
                        logging.info("MM D-G Sim: mm_dg_portfolio_state_df_sim is empty after run_loop.")

                except ValueError as ve_init_sim: # Catch specific init errors from MatrixDeltaGammaHedgeSimple
                    st.error(f"MM D-G Sim Initialization Error: {ve_init_sim}")
                    logging.error(f"MM D-G Sim: ValueError during MatrixDeltaGammaHedgeSimple initialization: {ve_init_sim}", exc_info=True)
                except Exception as e_mm_sim: # Catch any other errors during instantiation or run_loop
                    st.error(f"MM D-G Sim Runtime Error: {e_mm_sim}")
                    logging.error(f"MM D-G Sim: Exception during run or init: {e_mm_sim}", exc_info=True)
            
            else: # If gamma_hedger_candidate_df is still empty after fallbacks
                st.warning("MM D-G Sim: Could not find any suitable Call option in the latest snapshot to use as a gamma hedging instrument for the simulation.")
                logging.warning("MM D-G Sim: gamma_hedger_candidate_df remained empty. Cannot run simulation.")
        
        else: # Conditions for running simulation not met (e.g., dft or df_krak_5m empty)
            missing_data_reasons = []
            if dft.empty: missing_data_reasons.append("'dft' (historical options data) is empty")
            if df_krak_5m.empty: missing_data_reasons.append("'df_krak_5m' (historical spot data) is empty")
            if not selected_expiry: missing_data_reasons.append("'selected_expiry' is not set")
            if dft_latest.empty: missing_data_reasons.append("'dft_latest' (latest options snapshot) is empty")
            
            st.warning(f"MM D-G Sim: Cannot run simulation due to missing data: {', '.join(missing_data_reasons)}.")
            logging.warning(f"MM D-G Sim: Pre-conditions not met. Reasons: {', '.join(missing_data_reasons)}")



    gc.collect()
    logging.info(f"Focused Dashboard rendering complete for {coin} {e_str if e_str else ''}.")

if __name__ == "__main__":
    main()
