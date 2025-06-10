import streamlit as st
import datetime as dt
import scipy.stats as si
# import scipy.interpolate # Not directly needed if using fixed IV/Price for hedger
import pandas as pd
import requests
import numpy as np
import ccxt
# from toolz.curried import pipe, valmap, get_in, curry, valfilter # Not used
import plotly.express as px # Potentially used by helper plot
import plotly.graph_objects as go
from scipy.stats import norm
# from scipy.interpolate import interp1d # Not directly needed
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
    """Fetch historical data for a tuple of instruments."""
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
            except:  # Corrected indentation
                return np.nan

        def safe_get_type(s):
            try:
                return s.split('-')[-1]
            except:  # Corrected indentation
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

def fetch_kraken_data(coin="BTC", days=7): # Spot data needed for historical greeks
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


def compute_delta(row, S, snapshot_time_utc, r=0.0):
    try:
        k = row.get('k'); sigma = row.get('iv_close'); option_type = row.get('option_type')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(option_type) or option_type not in ['C', 'P'] or pd.isna(sigma) or pd.isna(r): return np.nan
        expiry_date = row.get('expiry_datetime_col') # Expect this to be pre-calculated and tz-aware UTC
        if pd.isna(expiry_date): return np.nan # Cannot proceed without valid expiry
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc) # Ensure snapshot is UTC
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
    try:
        k = row.get('k'); sigma = row.get('iv_close')
        if pd.isna(S) or S <= 1e-9 or pd.isna(k) or k <= 1e-9 or pd.isna(sigma) or pd.isna(r) or sigma < 1e-7: return np.nan if not (sigma < 1e-7) else 0.0
        expiry_date = row.get('expiry_datetime_col')
        if pd.isna(expiry_date): return np.nan
        if snapshot_time_utc.tzinfo is None: snapshot_time_utc = snapshot_time_utc.replace(tzinfo=dt.timezone.utc)
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

# --- MM Delta-Gamma Hedge Class and Plotting ---
class MatrixDeltaGammaHedgeSimple:
    def __init__(self, df_portfolio_options, spot_df, symbol="BTC", risk_free_rate=0.0, gamma_hedge_instrument_details=None):
        self.df_portfolio_options = df_portfolio_options.copy(); self.spot_df = spot_df.copy(); self.symbol = symbol.upper(); self.risk_free_rate = risk_free_rate; self.gamma_hedge_instrument_details = gamma_hedge_instrument_details
        self.portfolio_state_log = []; self.hedge_actions_log = []; self.current_underlying_hedge_qty = 0.0; self.current_gamma_option_hedge_qty = 0.0
        self._validate_inputs()

    def _validate_inputs(self):
        if self.symbol not in ["BTC", "ETH"]: raise ValueError(f"Incorrect symbol: {self.symbol}")
        req_cols = ['date_time', 'instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'mark_price_close', 'expiry_datetime_col']
        if self.df_portfolio_options.empty or not all(c in self.df_portfolio_options.columns for c in req_cols): raise ValueError(f"df_portfolio_options missing columns")
        for df in [self.df_portfolio_options, self.spot_df]:
            if not pd.api.types.is_datetime64_any_dtype(df['date_time']): df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
            elif df['date_time'].dt.tz is None: df['date_time'] = df['date_time'].dt.tz_localize('UTC')
            elif df['date_time'].dt.tz != dt.timezone.utc: df['date_time'] = df['date_time'].dt.tz_convert('UTC')
        if not pd.api.types.is_datetime64_any_dtype(self.df_portfolio_options['expiry_datetime_col']): self.df_portfolio_options['expiry_datetime_col'] = pd.to_datetime(self.df_portfolio_options['expiry_datetime_col'], utc=True)

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
        if not self.gamma_hedge_instrument_details: return np.nan, np.nan, np.nan
        details = self.gamma_hedge_instrument_details
        hedger_row = pd.Series({'instrument_name': details['name'], 'k': details['k'], 'option_type': details['option_type'], 'expiry_datetime_col': details['expiry_datetime_col'], 'iv_close': details['iv_close_source'](timestamp, spot_price) if callable(details['iv_close_source']) else details['iv_close_source']})
        if pd.isna(hedger_row['iv_close']) or hedger_row['iv_close'] <= 0: return np.nan, np.nan, np.nan
        hedger_delta = compute_delta(hedger_row, spot_price, timestamp, self.risk_free_rate); hedger_gamma = compute_gamma(hedger_row, spot_price, timestamp, self.risk_free_rate)
        hedger_price = details['mark_price_close_source'](timestamp, spot_price) if 'mark_price_close_source' in details and callable(details['mark_price_close_source']) else details.get('mark_price_close_source', np.nan)
        if pd.isna(hedger_delta) or pd.isna(hedger_gamma) or abs(hedger_gamma) < 1e-7: return np.nan, np.nan, hedger_price
        return hedger_delta, hedger_gamma, hedger_price

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
        if self.df_portfolio_options.empty or self.spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        latest_hist_ts = self.df_portfolio_options['date_time'].max(); latest_spot_ts = self.spot_df['date_time'].max()
        if pd.isna(latest_hist_ts) or pd.isna(latest_spot_ts): return pd.DataFrame(), pd.DataFrame()
        latest_timestamp = min(latest_hist_ts, latest_spot_ts); min_data_ts = max(self.df_portfolio_options['date_time'].min(), self.spot_df['date_time'].min())
        if pd.isna(min_data_ts): return pd.DataFrame(), pd.DataFrame()
        potential_start_timestamp = latest_timestamp - pd.Timedelta(days=days); start_timestamp = max(potential_start_timestamp, min_data_ts)
        if start_timestamp >= latest_timestamp : return pd.DataFrame(), pd.DataFrame()
        sim_options_df = self.df_portfolio_options[(self.df_portfolio_options['date_time'] >= start_timestamp) & (self.df_portfolio_options['date_time'] <= latest_timestamp)].copy()
        sim_spot_df = self.spot_df[(self.spot_df['date_time'] >= start_timestamp) & (self.spot_df['date_time'] <= latest_timestamp)].copy()
        if sim_options_df.empty or sim_spot_df.empty: return pd.DataFrame(), pd.DataFrame()
        loop_driving_timestamps = sorted(sim_options_df['date_time'].unique())
        if not loop_driving_timestamps: return pd.DataFrame(), pd.DataFrame()
        loop_timestamps_df = pd.DataFrame({'date_time': loop_driving_timestamps})
        spot_for_sim = pd.merge_asof(left=loop_timestamps_df.sort_values('date_time'), right=sim_spot_df[['date_time', 'close']].sort_values('date_time'), on='date_time', direction='backward', tolerance=pd.Timedelta('10min'))
        spot_for_sim['close'] = spot_for_sim['close'].ffill().bfill(); spot_for_sim = spot_for_sim.dropna(subset=['date_time', 'close'])
        if spot_for_sim.empty: return pd.DataFrame(), pd.DataFrame()
        final_sim_timestamps = sorted(spot_for_sim['date_time'].unique())
        if not final_sim_timestamps: return pd.DataFrame(), pd.DataFrame()
        self.portfolio_state_log = []; self.hedge_actions_log = []; self.current_underlying_hedge_qty = 0.0; self.current_gamma_option_hedge_qty = 0.0; trade_tolerance = 1e-6
        for ts in final_sim_timestamps:
            try:
                spot_price_at_ts = spot_for_sim.loc[spot_for_sim['date_time'] == ts, 'close'].iloc[0]
                if pd.isna(spot_price_at_ts) or spot_price_at_ts <= 0: continue
                port_val, port_delta, port_gamma = self._get_portfolio_greeks(ts, spot_price_at_ts)
                if pd.isna(port_delta) or pd.isna(port_gamma): continue
                hedger_D, hedger_G, hedger_P = np.nan, np.nan, np.nan
                if self.gamma_hedge_instrument_details:
                    hedger_D, hedger_G, hedger_P = self._get_gamma_hedger_greeks_and_price(ts, spot_price_at_ts)
                    if pd.isna(hedger_D) or pd.isna(hedger_G): hedger_D, hedger_G, hedger_P = 0.0, 1e-9, 0.0
                else: hedger_D, hedger_G, hedger_P = 0.0, 1e-9, 0.0
                target_B, target_n_underlying, target_n_gamma_opt = self._solve_delta_gamma_hedge_system(port_val, port_delta, port_gamma, spot_price_at_ts, hedger_D, hedger_G, hedger_P)
                if pd.notna(target_n_underlying):
                    trade_size_underlying = target_n_underlying - self.current_underlying_hedge_qty
                    if abs(trade_size_underlying) > trade_tolerance: self.hedge_actions_log.append({'timestamp': ts, 'instrument': self.symbol + '-PERP', 'action': 'buy' if trade_size_underlying > 0 else 'sell', 'size': abs(trade_size_underlying), 'price': spot_price_at_ts, 'type': 'delta_underlying'}); self.current_underlying_hedge_qty = target_n_underlying
                if self.gamma_hedge_instrument_details and pd.notna(target_n_gamma_opt):
                    trade_size_gamma_opt = target_n_gamma_opt - self.current_gamma_option_hedge_qty
                    if abs(trade_size_gamma_opt) > trade_tolerance: self.hedge_actions_log.append({'timestamp': ts, 'instrument': self.gamma_hedge_instrument_details['name'], 'action': 'buy' if trade_size_gamma_opt > 0 else 'sell', 'size': abs(trade_size_gamma_opt), 'price': hedger_P if pd.notna(hedger_P) else np.nan, 'type': 'gamma_option'}); self.current_gamma_option_hedge_qty = target_n_gamma_opt
                net_delta_final = port_delta + self.current_underlying_hedge_qty * 1.0 + (self.current_gamma_option_hedge_qty * hedger_D if pd.notna(hedger_D) else 0)
                net_gamma_final = port_gamma + (self.current_gamma_option_hedge_qty * hedger_G if pd.notna(hedger_G) else 0)
                self.portfolio_state_log.append({'timestamp': ts, 'spot_price': spot_price_at_ts, 'portfolio_value': port_val, 'portfolio_delta': port_delta, 'portfolio_gamma': port_gamma, 'target_B': target_B, 'target_n_underlying': target_n_underlying, 'target_n_gamma_opt': target_n_gamma_opt, 'current_n_underlying': self.current_underlying_hedge_qty, 'current_n_gamma_opt': self.current_gamma_option_hedge_qty, 'hedger_delta_at_ts': hedger_D, 'hedger_gamma_at_ts': hedger_G, 'hedger_price_at_ts': hedger_P, 'net_delta_final': net_delta_final, 'net_gamma_final': net_gamma_final})
            except Exception: self.portfolio_state_log.append({'timestamp': ts, 'spot_price': np.nan, 'portfolio_value':np.nan, 'portfolio_delta':np.nan, 'portfolio_gamma':np.nan, 'target_B':np.nan, 'target_n_underlying':np.nan, 'target_n_gamma_opt':np.nan, 'current_n_underlying':self.current_underlying_hedge_qty, 'current_n_gamma_opt':self.current_gamma_option_hedge_qty, 'hedger_delta_at_ts':np.nan, 'hedger_gamma_at_ts':np.nan, 'hedger_price_at_ts':np.nan, 'net_delta_final':np.nan, 'net_gamma_final':np.nan})
        return pd.DataFrame(self.portfolio_state_log), pd.DataFrame(self.hedge_actions_log)

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

def display_mm_gamma_adjustment_analysis(dft_latest_snap, spot_price, snapshot_time_utc, risk_free_rate=0.0):
    st.subheader("MM Indicative Delta-Gamma Hedge Adjustment (Selected Expiry)")
    st.caption("Assumes Market Maker is short the entire displayed option book for this expiry. Shows theoretical adjustment using a near-ATM call from the same expiry to hedge gamma.")
    required_cols = ['instrument_name', 'k', 'option_type', 'iv_close', 'open_interest', 'expiry_datetime_col']
    if dft_latest_snap.empty or not all(c in dft_latest_snap.columns for c in required_cols) or pd.isna(spot_price) or spot_price <= 0: return
    df_book = dft_latest_snap.copy(); df_book['open_interest'] = pd.to_numeric(df_book['open_interest'], errors='coerce').fillna(0); df_book = df_book[df_book['open_interest'] > 0]
    if df_book.empty: return
    df_book['mm_delta_pos'] = -1 * df_book.apply(lambda r: compute_delta(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1) * df_book['open_interest']
    df_book['mm_gamma_pos'] = -1 * df_book.apply(lambda r: compute_gamma(r, spot_price, snapshot_time_utc, risk_free_rate), axis=1) * df_book['open_interest']
    mm_net_delta_initial = df_book['mm_delta_pos'].sum(skipna=True); mm_net_gamma_initial = df_book['mm_gamma_pos'].sum(skipna=True)
    if pd.isna(mm_net_delta_initial) or pd.isna(mm_net_gamma_initial): return
    st.metric("MM Initial Net Delta (Book)", f"{mm_net_delta_initial:,.2f}"); st.metric("MM Initial Net Gamma (Book)", f"{mm_net_gamma_initial:,.4f}")
    gamma_hedger_selected = None; all_calls_in_book = df_book[df_book['option_type'] == 'C'].copy()
    if not all_calls_in_book.empty:
        all_calls_in_book['moneyness_dist'] = abs(all_calls_in_book['k'] - spot_price)
        atm_ish_calls = all_calls_in_book[all_calls_in_book['k'] >= spot_price].sort_values('moneyness_dist')
        gamma_hedger_selected_name = atm_ish_calls['instrument_name'].iloc[0] if not atm_ish_calls.empty else (all_calls_in_book.sort_values('moneyness_dist')['instrument_name'].iloc[0] if not all_calls_in_book.empty else None)
        if gamma_hedger_selected_name:
            gamma_hedger_selected_row_data = df_book[df_book['instrument_name'] == gamma_hedger_selected_name].iloc[[0]]
            if not gamma_hedger_selected_row_data.empty: gamma_hedger_selected = gamma_hedger_selected_row_data.iloc[0].copy()
    if gamma_hedger_selected is None: return
    hedger_details_name = gamma_hedger_selected['instrument_name']; st.info(f"Selected Gamma Hedging Instrument: {hedger_details_name}")
    D_h = compute_delta(gamma_hedger_selected, spot_price, snapshot_time_utc, risk_free_rate); G_h = compute_gamma(gamma_hedger_selected, spot_price, snapshot_time_utc, risk_free_rate)
    if pd.isna(D_h) or pd.isna(G_h) or abs(G_h) < 1e-7: return
    N_h = -mm_net_gamma_initial / G_h; delta_from_gamma_hedge = N_h * D_h; mm_net_delta_post_gamma_hedge = mm_net_delta_initial + delta_from_gamma_hedge; underlying_hedge_qty = -mm_net_delta_post_gamma_hedge
    st.markdown("#### Indicative Gamma Hedge Adjustment:"); cols_gamma_hedge = st.columns(3)
    with cols_gamma_hedge[0]: st.metric("Gamma Hedger Delta (Dₕ)", f"{D_h:.4f}")
    with cols_gamma_hedge[1]: st.metric("Gamma Hedger Gamma (Gₕ)", f"{G_h:.6f}")
    with cols_gamma_hedge[2]: action_gh = "Buy" if N_h > 0 else "Sell" if N_h < 0 else "Hold"; st.metric(f"Hedge Option Qty ({action_gh})", f"{abs(N_h):,.2f} units")
    st.metric("Delta Change from Gamma Hedge", f"{delta_from_gamma_hedge:,.2f}")
    st.markdown("---"); st.markdown("#### Indicative Final Delta Hedge (Post-Gamma Adj.):"); st.metric("MM Net Delta (After Gamma Hedge)", f"{mm_net_delta_post_gamma_hedge:,.2f}")
    action_underlying = "Buy" if underlying_hedge_qty > 0 else "Sell" if underlying_hedge_qty < 0 else "Hold"; st.metric(f"Final Underlying Hedge ({action_underlying} Spot/Perp)", f"{abs(underlying_hedge_qty):,.2f} {st.session_state.selected_coin}")
    final_net_delta_book = mm_net_delta_post_gamma_hedge + underlying_hedge_qty; st.success(f"**Resulting Book Net Delta (Post-All Adjustments):** {final_net_delta_book:,.4f}")


# --- Main Function Definition ---
def main(): # Make sure this is defined at the top level
    st.set_page_config(layout="wide", page_title="Delta Hedging & MM Focus")
    login()

    if 'selected_coin' not in st.session_state: st.session_state.selected_coin = "BTC"
    if 'snapshot_time' not in st.session_state: st.session_state.snapshot_time = dt.datetime.now(dt.timezone.utc)
    if 'risk_free_rate_input' not in st.session_state: st.session_state.risk_free_rate_input = 0.01

    st.title(f"{st.session_state.selected_coin} Options: Delta Hedging & MM Perspective")

    if st.sidebar.button("Logout"):
        keys_to_clear = [k for k in st.session_state.keys() if k != 'logged_in']
        for key in keys_to_clear: del st.session_state[key]
        st.session_state.logged_in = False; st.rerun()

    dft = pd.DataFrame(); dft_latest = pd.DataFrame(); ticker_data = {};
    df_krak_5m = pd.DataFrame(); df_krak_daily = pd.DataFrame() # For Market Memory
    all_instruments_list = []; expiry_options = []; all_instr_selected_expiry = []
    selected_expiry = None; e_str = ""; spot_price = np.nan
    all_calls_expiry = []; all_puts_expiry = []


    st.sidebar.header("Configuration")
    coin_options = ["BTC", "ETH"]
    if st.session_state.selected_coin not in coin_options: st.session_state.selected_coin = "BTC"
    current_coin_index = coin_options.index(st.session_state.selected_coin)
    selected_coin_from_widget = st.sidebar.selectbox("Select Cryptocurrency", coin_options, index=current_coin_index, key='coin_selector_mm_focus')
    if selected_coin_from_widget != st.session_state.selected_coin:
        st.session_state.selected_coin = selected_coin_from_widget; st.rerun()
    coin = st.session_state.selected_coin
    st.session_state.risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate", value=st.session_state.get('risk_free_rate_input', 0.01), min_value=0.0, max_value=0.2, step=0.001, format="%.3f", key="rf_rate_mm_focus")
    risk_free_rate = st.session_state.risk_free_rate_input

    with st.spinner("Fetching Thalex instruments..."): all_instruments_list = fetch_instruments()
    if not all_instruments_list: st.error("Failed to fetch Thalex instrument list."); st.stop()
    now_utc = dt.datetime.now(dt.timezone.utc)
    with st.spinner("Determining expiries..."): expiry_options = get_valid_expiration_options(now_utc)
    if not expiry_options: st.error(f"No valid future expiries for {coin}."); st.stop()
    
    default_expiry_idx = 0
    if expiry_options:
        for i, exp_dt in enumerate(expiry_options):
            if get_option_instruments(all_instruments_list, "C", exp_dt.strftime("%d%b%y").upper(), coin):
                default_expiry_idx = i; break
    selected_expiry = st.sidebar.selectbox("Select Expiry (MM & Strike Plots)", options=expiry_options, format_func=lambda dt_obj: dt_obj.strftime("%d %b %Y"), index=default_expiry_idx, key=f"expiry_selector_mm_focus_{coin}")
    if selected_expiry: e_str = selected_expiry.strftime("%d%b%y").upper()
    else: st.error("Please select an expiry date."); st.stop()

    all_calls_expiry = get_option_instruments(all_instruments_list, "C", e_str, coin)
    all_puts_expiry = get_option_instruments(all_instruments_list, "P", e_str, coin)
    all_instr_selected_expiry = sorted(all_calls_expiry + all_puts_expiry)

    with st.spinner(f"Fetching Kraken {coin} spot data..."):
        df_krak_5m = fetch_kraken_data(coin=coin, days=7)
        df_krak_daily = fetch_kraken_data_daily(days=365, coin=coin) # For market memory
    if df_krak_5m.empty or df_krak_daily.empty: st.error(f"Failed to fetch Kraken {coin} data."); st.stop()
    spot_price = df_krak_5m["close"].iloc[-1] if not df_krak_5m.empty else np.nan
    if pd.isna(spot_price): st.error("Could not determine latest spot price."); st.stop()
    
    st.header(f"MM & Delta Hedge Analysis for {coin} | Expiry: {selected_expiry.strftime('%d %b %Y')} | Spot: ${spot_price:,.2f}")
    st.markdown(f"*Snapshot Time (UTC): {st.session_state.snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}*")

    if not all_instr_selected_expiry: st.error(f"No {coin} options for selected expiry {e_str}."); st.stop()
    
    with st.spinner(f"Fetching historical options data for expiry {e_str}..."):
        dft = fetch_data(tuple(all_instr_selected_expiry))
    
    required_cols_dft = ['date_time', 'instrument_name', 'k', 'option_type', 'mark_price_close', 'iv_close', 'expiry_datetime_col']
    if dft.empty or not all(col in dft.columns for col in required_cols_dft): st.error(f"Failed to fetch/process options for {e_str}."); st.stop()
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
        else: dft["delta"] = np.nan; dft["gamma"] = np.nan # Fallback
    else: dft["delta"] = np.nan; dft["gamma"] = np.nan

    dft_latest = pd.DataFrame()
    if not dft.empty and 'date_time' in dft.columns:
        try:
            latest_indices = dft.groupby('instrument_name')['date_time'].idxmax()
            dft_latest_temp = dft.loc[latest_indices].copy()
            greek_cols_check = ['instrument_name', 'k', 'iv_close', 'option_type', 'expiry_datetime_col']
            if all(c in dft_latest_temp.columns for c in greek_cols_check):
                current_time = st.session_state.snapshot_time
                dft_latest_temp["delta"] = dft_latest_temp.apply(lambda r: compute_delta(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["gamma"] = dft_latest_temp.apply(lambda r: compute_gamma(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["vega"] = dft_latest_temp.apply(lambda r: compute_vega(r, spot_price, current_time), axis=1).astype('float32')
                dft_latest_temp["charm"] = dft_latest_temp.apply(lambda r: compute_charm(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                dft_latest_temp["vanna"] = dft_latest_temp.apply(lambda r: compute_vanna(r, spot_price, current_time, risk_free_rate), axis=1).astype('float32')
                if 'gamma' in dft_latest_temp.columns and 'open_interest' in dft_latest_temp.columns:
                    dft_latest_temp["gex"] = dft_latest_temp.apply(lambda r: compute_gex(r, spot_price, r['open_interest']), axis=1).astype('float32')
                else: dft_latest_temp["gex"] = np.nan
                dft_latest = dft_latest_temp
        except Exception: dft_latest = pd.DataFrame()

    def safe_plot(plot_func, *args, **kwargs):
        try:
            if callable(plot_func): plot_func(*args, **kwargs)
        except Exception as e: st.error(f"Plot error in '{getattr(plot_func, '__name__', 'N/A')}'. Check logs."); logging.error(f"Plot error", exc_info=True)

    # =========================================================================
    # 1. Understanding Key Metrics & Market Memory
    # =========================================================================
    st.markdown("---"); st.header("Key Metrics & Market Memory")
    # ... (Volatility Snapshot, Hurst, Autocorrelation plots as before) ...
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
        st.subheader("Hurst Exponent (Market Memory - Classic R/S)")
        if not df_krak_daily.empty and 'close' in df_krak_daily:
            daily_log_returns_hurst = np.log(df_krak_daily['close'] / df_krak_daily['close'].shift(1)) # Recalculate for safety
            hurst_val, hurst_data_df = calculate_hurst_lo_modified(daily_log_returns_hurst) # Using classic R/S (q_method default will be effectively 0)
            safe_plot(plot_hurst_exponent, hurst_val, hurst_data_df)
        st.markdown("---"); st.subheader("Autocorrelation (Market Memory)")
        if not df_krak_daily.empty and 'close' in df_krak_daily and 'daily_log_returns_hurst' in locals():
            safe_plot(calculate_and_display_autocorrelation, daily_log_returns_hurst, windows=[7, 15, 30])


    # =========================================================================
    # 2. Market Maker Positioning
    # =========================================================================
    st.markdown("---"); st.header(f"Market Maker Positioning (Expiry: {selected_expiry.strftime('%d%b%y')})")
    # ... (MM Greek exposure metrics, heatmaps, strike views as before) ...
    net_delta_mm = np.nan; net_gex_mm = np.nan; net_vega_mm = np.nan; net_vanna_mm = np.nan; net_charm_mm = np.nan
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

    # =========================================================================
    # 3. MM Indicative Delta-Gamma Hedge Adjustment
    # =========================================================================
    st.markdown("---"); st.header("MM Indicative Delta-Gamma Hedge Adjustment")
    if not dft_latest.empty and pd.notna(spot_price) and pd.notna(st.session_state.snapshot_time):
        safe_plot(display_mm_gamma_adjustment_analysis, dft_latest, spot_price, st.session_state.snapshot_time, risk_free_rate)
    
    # --- MM Delta-Gamma Hedge Sim (Plot Only) ---
    show_mm_dg_sim_main = st.sidebar.checkbox("Show MM D-G Hedge Sim Plot", value=True, key="show_mm_dg_sim_main_plot")
    if show_mm_dg_sim_main:
        if not dft.empty and not df_krak_5m.empty and selected_expiry and not dft_latest.empty:
            gamma_hedger_candidate_df = dft_latest[(dft_latest['option_type'] == 'C') & (dft_latest['k'] >= spot_price)].sort_values('k')
            if gamma_hedger_candidate_df.empty: gamma_hedger_candidate_df = dft_latest[dft_latest['option_type'] == 'C'].sort_values(by=lambda x_series: abs(x_series - spot_price) if isinstance(x_series, pd.Series) and pd.api.types.is_numeric_dtype(x_series) else np.inf)

            if not gamma_hedger_candidate_df.empty:
                gamma_hedger_row_sim = gamma_hedger_candidate_df.iloc[0]
                gamma_hedger_details_sim = {
                    'name': gamma_hedger_row_sim['instrument_name'], 'k': gamma_hedger_row_sim['k'], 'option_type': 'C',
                    'expiry_datetime_col': gamma_hedger_row_sim['expiry_datetime_col'],
                    'iv_close_source': gamma_hedger_row_sim['iv_close'], # Use fixed IV from latest snapshot for simplicity
                    'mark_price_close_source': gamma_hedger_row_sim['mark_price_close'] # Use fixed price
                }
                try:
                    mm_dg_sim_instance = MatrixDeltaGammaHedgeSimple(df_portfolio_options=dft, spot_df=df_krak_5m, symbol=coin, risk_free_rate=risk_free_rate, gamma_hedge_instrument_details=gamma_hedger_details_sim)
                    with st.spinner("Running MM Delta-Gamma Hedge Simulation..."):
                        mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim = mm_dg_sim_instance.run_loop(days=5)
                    if not mm_dg_portfolio_state_df_sim.empty:
                        safe_plot(plot_mm_delta_gamma_hedge, mm_dg_portfolio_state_df_sim, mm_dg_hedge_actions_df_sim, coin)
                    else: st.warning("MM Delta-Gamma Hedge Simulation produced no state data to plot.")
                except Exception as e_mm_sim: st.error(f"MM D-G Sim Error: {e_mm_sim}")
            else: st.warning("Could not find a Call option in latest snapshot for MM D-G Sim.")

    # =========================================================================
    # 4. ITM Gamma Exposure Analysis
    # =========================================================================
    st.markdown("---"); st.header(f"ITM Gamma Exposure Analysis (Expiry: {selected_expiry.strftime('%d%b%y') if selected_expiry else 'N/A'})")
    if not dft.empty and not df_krak_5m.empty and pd.notna(spot_price) and selected_expiry:
        safe_plot(compute_and_plot_itm_gex_ratio, dft=dft, df_krak_5m=df_krak_5m, spot_price_latest=spot_price, selected_expiry_obj=selected_expiry)

    # =========================================================================
    # 5. Delta Hedging Simulations (Traditional and Matrix)
    # =========================================================================
    st.markdown("---"); st.header("Delta Hedging Simulations (Full Expiry Book)")
    show_delta_hedging_sims_main = st.sidebar.checkbox("Show Full Book Delta Hedge Sims", value=True, key="show_delta_hedging_main_plot_v2") # Changed Key
    
    if show_delta_hedging_sims_main:
        st.sidebar.markdown("##### Full Book Delta Hedge Params")
        # These will apply to the HedgeThalex traditional sim
        use_dynamic_threshold_main = st.sidebar.checkbox("Dynamic Threshold?", value=False, key="use_dyn_thresh_main_plot_v2")
        use_dynamic_hedge_size_main = st.sidebar.checkbox("Dynamic Hedge Size?", value=False, key="use_dyn_size_main_plot_v2")
        base_thresh_main = st.sidebar.slider("Base Δ Threshold", 0.01, 0.5, 0.20, 0.01, key='base_thresh_slider_main_plot_v2', format="%.2f")
        min_hedge_rat_main = st.sidebar.slider("Min Hedge Ratio (%)", 0, 100, 50, 5, key='min_hedge_ratio_slider_main_plot_v2', format="%d%%", disabled=not use_dynamic_hedge_size_main) / 100.0
        max_hedge_rat_main = st.sidebar.slider("Max Hedge Ratio (%)", 0, 100, 100, 5, key='max_hedge_ratio_slider_main_plot_v2', format="%d%%", disabled=not use_dynamic_hedge_size_main) / 100.0
        
        if not dft.empty and not df_krak_5m.empty:
            try:
                hedge_instance_main = HedgeThalex(df_historical=dft, spot_df=df_krak_5m, symbol=coin, base_threshold=base_thresh_main, use_dynamic_threshold=use_dynamic_threshold_main, use_dynamic_hedge_size=use_dynamic_hedge_size_main, min_hedge_ratio=min_hedge_rat_main, max_hedge_ratio=max_hedge_rat_main)
                with st.spinner("Running Traditional Delta Hedge Sim..."):
                    delta_df_main, hedge_df_main = hedge_instance_main.run_loop(days=5)
                if not delta_df_main.empty: safe_plot(plot_delta_hedging_thalex, delta_df_main, hedge_df_main, base_thresh_main, use_dynamic_threshold_main, coin, spot_price, df_krak_5m)
            except Exception as e: st.error(f"Trad. Hedging Sim Error (Main): {e}")
        
        if not dft.empty and not df_krak_5m.empty:
            try:
                matrix_hedge_instance_main = MatrixHedgeThalex(df_historical=dft, spot_df=df_krak_5m, symbol=coin)
                with st.spinner("Running Matrix Delta Hedge Sim..."):
                    matrix_portfolio_state_df_main, matrix_hedge_actions_df_main = matrix_hedge_instance_main.run_loop(days=5)
                if not matrix_portfolio_state_df_main.empty: safe_plot(plot_matrix_hedge_thalex, matrix_portfolio_state_df_main, matrix_hedge_actions_df_main, coin)
            except Exception as e: st.error(f"Matrix Hedging Sim Error (Main): {e}")

    # =========================================================================
    # 6. Options Premium Bias Comparison
    # =========================================================================
    st.markdown("---"); st.header(f"Options Premium Bias Comparison (Expiry: {selected_expiry.strftime('%d%b%y')})")
    df_atm_results_bias = calculate_atm_premium_data(dft, df_krak_5m, selected_expiry)
    df_itm_results_bias = calculate_itm_premium_data(dft, df_krak_5m, selected_expiry)
    safe_plot(plot_combined_premium_difference, df_atm_results_bias, df_itm_results_bias, selected_expiry.strftime('%d%b%y'))

    # =========================================================================
    # 7. Delta Neutral Pair Analysis (using "ideal" pair)
    # =========================================================================
    st.markdown("---"); st.header("Delta Neutral Pair Analysis (Ideal Pair)")
    st.sidebar.markdown("---"); st.sidebar.subheader("Ideal Pair Delta Hedge Sim")
    find_ideal_pair_button = st.sidebar.button("Find Ideal Pair for Sim", key="find_ideal_pair_btn_deltafocus_v3") # Changed Key

    # Initialize ideal pair variables outside the button click
    selected_call_instr_ideal = None
    selected_put_instr_ideal = None

    if find_ideal_pair_button and not dft_latest.empty:
        with st.spinner("Finding ideal OTM pair..."):
            if 'delta' not in dft_latest.columns: st.sidebar.error("Deltas not on latest snapshot.")
            else:
                calls_latest_ideal = dft_latest[dft_latest['option_type'] == 'C'].copy(); puts_latest_ideal = dft_latest[dft_latest['option_type'] == 'P'].copy()
                target_call_delta = 0.25; target_put_delta = -0.25
                if not calls_latest_ideal.empty: calls_latest_ideal['delta_diff'] = abs(calls_latest_ideal['delta'] - target_call_delta); selected_call_instr_ideal = calls_latest_ideal.loc[calls_latest_ideal['delta_diff'].idxmin(), 'instrument_name'] if not calls_latest_ideal['delta_diff'].empty else None
                if not puts_latest_ideal.empty: puts_latest_ideal['delta_diff'] = abs(puts_latest_ideal['delta'] - target_put_delta); selected_put_instr_ideal = puts_latest_ideal.loc[puts_latest_ideal['delta_diff'].idxmin(), 'instrument_name'] if not puts_latest_ideal['delta_diff'].empty else None
                if selected_call_instr_ideal and selected_put_instr_ideal: st.sidebar.success(f"Ideal Pair Found.") # Simplified message
    
    # Fallback if button not pressed or ideal pair not found
    if not selected_call_instr_ideal and all_calls_expiry: selected_call_instr_ideal = all_calls_expiry[0]
    if not selected_put_instr_ideal and all_puts_expiry: selected_put_instr_ideal = all_puts_expiry[0]

    if selected_call_instr_ideal and selected_put_instr_ideal:
        st.caption(f"Using Pair for Delta Neutral Sim: Call: {selected_call_instr_ideal} | Put: {selected_put_instr_ideal}")
        safe_plot(plot_net_delta_otm_pair, dft=dft, df_spot_hist=df_krak_5m, exchange_instance=exchange1, selected_call_instr=selected_call_instr_ideal, selected_put_instr=selected_put_instr_ideal)
    else:
        st.warning("Could not determine a call/put pair for Delta Neutral Pair Analysis.")

    gc.collect()
    logging.info(f"Focused Dashboard rendering complete for {coin} {e_str if e_str else ''}.")

if __name__ == "__main__":
    main()
