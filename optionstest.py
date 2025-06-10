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
    if not dft.empty and not df_krak_5m.empty and selected_expiry:
        safe_plot(plot_delta_oi_heatmap_refined, dft, df_krak_5m, selected_expiry)
        safe_plot(plot_gex_heatmap, dft, df_krak_5m, selected_expiry)
        safe_plot(plot_net_delta_flow_heatmap, dft, df_krak_5m, selected_expiry, coin)

    # =========================== MM Indicative Delta-Gamma Hedge ==========================
    st.markdown("---"); st.header("MM Indicative Delta-Gamma Hedge Adjustment")
    if not dft_latest.empty and pd.notna(spot_price) and pd.notna(st.session_state.snapshot_time):
        safe_plot(display_mm_gamma_adjustment_analysis, dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)
    
    show_mm_dg_sim_main = st.sidebar.checkbox("Show MM D-G Hedge Sim Plot", value=True, key="show_mm_dg_sim_main_plot_v2") # Changed key
    if show_mm_dg_sim_main:
        if not dft.empty and not df_krak_5m.empty and selected_expiry and not dft_latest.empty:
            gamma_hedger_candidate_df = dft_latest[(dft_latest['option_type'] == 'C') & (dft_latest['k'] >= spot_price)].sort_values('k')
            if gamma_hedger_candidate_df.empty and not dft_latest[dft_latest['option_type'] == 'C'].empty: # Fallback if no OTM/ATM calls
                 gamma_hedger_candidate_df = dft_latest[dft_latest['option_type'] == 'C'].iloc[[abs(dft_latest[dft_latest['option_type'] == 'C']['k'] - spot_price).idxmin()]]


            if not gamma_hedger_candidate_df.empty:
                gamma_hedger_row_sim = gamma_hedger_candidate_df.iloc[0]
                gamma_hedger_details_sim = {
                    'name': gamma_hedger_row_sim['instrument_name'], 'k': gamma_hedger_row_sim['k'], 'option_type': 'C',
                    'expiry_datetime_col': gamma_hedger_row_sim['expiry_datetime_col'],
                    'iv_close_source': gamma_hedger_row_sim['iv_close'],
                    'mark_price_close_source': gamma_hedger_row_sim['mark_price_close']
                }
                try:
                    mm_dg_sim_instance = MatrixDeltaGammaHedgeSimple(df_portfolio_options=dft, spot_df=df_krak_5m, symbol=coin, risk_free_rate=risk_free_rate, gamma_hedge_instrument_details=gamma_hedger_details_sim)
                    with st.spinner("Running MM Delta-Gamma Hedge Simulation..."):
                        mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim = mm_dg_sim_instance.run_loop(days=5)
                    if not mm_dg_portfolio_state_df_sim.empty:
                        safe_plot(plot_mm_delta_gamma_hedge, mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim, coin)
                except Exception as e_mm_sim: st.error(f"MM D-G Sim Error: {e_mm_sim}")
            else: st.warning("Could not find a suitable Call option in latest snapshot for MM D-G Sim.")


    # =========================== ITM Gamma Exposure Analysis ==========================
    st.markdown("---"); st.header(f"ITM Gamma Exposure Analysis (Expiry: {selected_expiry.strftime('%d%b%y') if selected_expiry else 'N/A'})")
    if not dft.empty and not df_krak_5m.empty and pd.notna(spot_price) and selected_expiry:
        safe_plot(compute_and_plot_itm_gex_ratio, dft=dft, df_krak_5m=df_krak_5m, spot_price_latest=spot_price, selected_expiry_obj=selected_expiry)

    # =========================== Delta Hedging Simulations ==========================
    st.markdown("---"); st.header("Delta Hedging Simulations (Full Expiry Book)")
    show_delta_hedging_sims_main = st.sidebar.checkbox("Show Full Book Delta Hedge Sims", value=True, key="show_delta_hedging_main_plot_v3") # Changed Key
    if show_delta_hedging_sims_main:
        st.sidebar.markdown("##### Full Book Delta Hedge Params")
        use_dynamic_threshold_main = st.sidebar.checkbox("Dynamic Threshold?", value=False, key="use_dyn_thresh_main_plot_v3")
        use_dynamic_hedge_size_main = st.sidebar.checkbox("Dynamic Hedge Size?", value=False, key="use_dyn_size_main_plot_v3")
        base_thresh_main = st.sidebar.slider("Base Δ Threshold", 0.01, 0.5, 0.20, 0.01, key='base_thresh_slider_main_plot_v3', format="%.2f")
        min_hedge_rat_main = st.sidebar.slider("Min Hedge Ratio (%)", 0, 100, 50, 5, key='min_hedge_ratio_slider_main_plot_v3', format="%d%%", disabled=not use_dynamic_hedge_size_main) / 100.0
        max_hedge_rat_main = st.sidebar.slider("Max Hedge Ratio (%)", 0, 100, 100, 5, key='max_hedge_ratio_slider_main_plot_v3', format="%d%%", disabled=not use_dynamic_hedge_size_main) / 100.0
        if not dft.empty and not df_krak_5m.empty:
            try:
                hedge_instance_main = HedgeThalex(df_historical=dft, spot_df=df_krak_5m, symbol=coin, base_threshold=base_thresh_main, use_dynamic_threshold=use_dynamic_threshold_main, use_dynamic_hedge_size=use_dynamic_hedge_size_main, min_hedge_ratio=min_hedge_rat_main, max_hedge_ratio=max_hedge_rat_main)
                with st.spinner("Running Traditional Delta Hedge Sim..."): delta_df_main, hedge_df_main = hedge_instance_main.run_loop(days=5)
                if not delta_df_main.empty: safe_plot(plot_delta_hedging_thalex, delta_df_main, hedge_df_main, base_thresh_main, use_dynamic_threshold_main, coin, spot_price, df_krak_5m)
            except Exception as e: st.error(f"Trad. Hedging Sim Error (Main): {e}")
        if not dft.empty and not df_krak_5m.empty:
            try:
                matrix_hedge_instance_main = MatrixHedgeThalex(df_historical=dft, spot_df=df_krak_5m, symbol=coin)
                with st.spinner("Running Matrix Delta Hedge Sim..."): matrix_portfolio_state_df_main, matrix_hedge_actions_df_main = matrix_hedge_instance_main.run_loop(days=5)
                if not matrix_portfolio_state_df_main.empty: safe_plot(plot_matrix_hedge_thalex, matrix_portfolio_state_df_main, matrix_hedge_actions_df_main, coin)
            except Exception as e: st.error(f"Matrix Hedging Sim Error (Main): {e}")

    # =========================== Options Premium Bias Comparison ==========================
    st.markdown("---"); st.header(f"Options Premium Bias Comparison (Expiry: {selected_expiry.strftime('%d%b%y')})")
    df_atm_results_bias = calculate_atm_premium_data(dft, df_krak_5m, selected_expiry)
    df_itm_results_bias = calculate_itm_premium_data(dft, df_krak_5m, selected_expiry)
    safe_plot(plot_combined_premium_difference, df_atm_results_bias, df_itm_results_bias, selected_expiry.strftime('%d%b%y'))

    # =========================== Delta Neutral Pair Analysis ==========================
    st.markdown("---"); st.header("Delta Neutral Pair Analysis (Ideal Pair)")
    st.sidebar.markdown("---"); st.sidebar.subheader("Ideal Pair Delta Hedge Sim")
    find_ideal_pair_button = st.sidebar.button("Find Ideal Pair for Sim", key="find_ideal_pair_btn_deltafocus_v3")
    selected_call_instr_ideal = None; selected_put_instr_ideal = None
    if find_ideal_pair_button and not dft_latest.empty:
        with st.spinner("Finding ideal OTM pair..."):
            if 'delta' in dft_latest.columns:
                calls_latest_ideal = dft_latest[dft_latest['option_type'] == 'C'].copy(); puts_latest_ideal = dft_latest[dft_latest['option_type'] == 'P'].copy()
                target_call_delta = 0.25; target_put_delta = -0.25
                if not calls_latest_ideal.empty: calls_latest_ideal['delta_diff'] = abs(calls_latest_ideal['delta'] - target_call_delta); selected_call_instr_ideal = calls_latest_ideal.loc[calls_latest_ideal['delta_diff'].idxmin(), 'instrument_name'] if not calls_latest_ideal['delta_diff'].empty else None
                if not puts_latest_ideal.empty: puts_latest_ideal['delta_diff'] = abs(puts_latest_ideal['delta'] - target_put_delta); selected_put_instr_ideal = puts_latest_ideal.loc[puts_latest_ideal['delta_diff'].idxmin(), 'instrument_name'] if not puts_latest_ideal['delta_diff'].empty else None
                if selected_call_instr_ideal and selected_put_instr_ideal: st.sidebar.success(f"Ideal Pair Found.")
    if not selected_call_instr_ideal and all_calls_expiry: selected_call_instr_ideal = all_calls_expiry[0]
    if not selected_put_instr_ideal and all_puts_expiry: selected_put_instr_ideal = all_puts_expiry[0]
    if selected_call_instr_ideal and selected_put_instr_ideal:
        st.caption(f"Using Pair for Delta Neutral Sim: Call: {selected_call_instr_ideal} | Put: {selected_put_instr_ideal}")
        safe_plot(plot_net_delta_otm_pair, dft=dft, df_spot_hist=df_krak_5m, exchange_instance=exchange1, selected_call_instr=selected_call_instr_ideal, selected_put_instr=selected_put_instr_ideal)
    else: st.warning("Could not determine a call/put pair for Delta Neutral Pair Analysis.")

    gc.collect()
    logging.info(f"Focused Dashboard rendering complete for {coin} {e_str if e_str else ''}.")

if __name__ == "__main__":
    main()
