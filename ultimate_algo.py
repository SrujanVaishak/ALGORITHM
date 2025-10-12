# ✅ ULTIMATE MASTER OPTION ALGO WITH INSTITUTIONAL LAYERS + ALL INDICES PARALLEL
# ✅ NIFTY + BANKNIFTY + SENSEX + FINNIFTY + MIDCPNIFTY + EICHERMOT + TRENT + RELIANCE
# ✅ Institutional Flow + OI/Delta Layers + Liquidity Hunting + Multi-timeframe + Telegram
# ✅ All indices running simultaneously in parallel threads

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading

warnings.filterwarnings("ignore")

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    # Get current time in IST (UTC +5:30)
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    
    # Check if current IST time is between 9:15 AM and 3:30 PM
    market_open = dtime(9,15) <= current_time_ist <= dtime(15,30)
    
    return market_open

# --------- AUTO STOP AT 3:30 PM IST ---------
def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR ALL INDICES ---------
def round_strike(index, price):
    """
    Safely round a price to nearest strike step per index.
    Returns None if price is None or NaN.
    """
    try:
        if price is None:
            return None
        # handle pandas NA / numpy nan
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        if index == "NIFTY": return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": return int(round(price / 100.0) * 100)
        elif index == "SENSEX": return int(round(price / 100.0) * 100)
        elif index == "FINNIFTY": return int(round(price / 50.0) * 50)
        elif index == "MIDCPNIFTY": return int(round(price / 25.0) * 25)
        elif index == "EICHERMOT": return int(round(price / 50.0) * 50)
        elif index == "TRENT": return int(round(price / 100.0) * 100)
        elif index == "RELIANCE": return int(round(price / 10.0) * 10)
        else: return int(round(price / 50.0) * 50)  # default
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR ALL INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", 
        "EICHERMOT": "EICHERMOT.NS",
        "TRENT": "TRENT.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        # Include both NFO and BFO for SENSEX
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            # Determine exchange based on symbol
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    # fallback if not enough history
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    # ensure numeric fallback
    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT (robust) ---------
def institutional_liquidity_hunt(index, df):
    """
    Calculates pre-market and opening range liquidity zones.
    Returns dual-sided liquidity zones for CE/PE trades.
    """
    # Pre-market / Overnight levels
    prev_high = None
    prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None
        prev_low = None

    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

    # Options OI zones (dummy placeholder) — guard against NaN last close
    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None

    if last_close_val is None:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    """
    Returns which side (CE/PE/BOTH) is appropriate based on price near liquidity zones.
    Robust to None zones and NaN price.
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:  # Threshold 5 points
                return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "PE"
        except:
            continue

    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    return None

# --------- STRATEGY CHECK ---------
def analyze_index_signal(index):
    df5=fetch_index_data(index,"5m","2d")
    df15=fetch_index_data(index,"15m","10d")
    if df5 is None or df15 is None:
        return None

    close5=ensure_series(df5["Close"])
    close15=ensure_series(df15["Close"])

    # guard - ensure we have enough data and non-NaN
    if len(close5) < 2 or len(close15) < 2:
        return None
    if close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    ema9_5=float(ta.trend.EMAIndicator(close5,9).ema_indicator().iloc[-1])
    ema21_5=float(ta.trend.EMAIndicator(close5,21).ema_indicator().iloc[-1])
    rsi5=float(ta.momentum.RSIIndicator(close5,14).rsi().iloc[-1])

    vol5 = ensure_series(df5["Volume"])
    vol_latest=float(vol5.iloc[-1])
    vol_avg=float(vol5.rolling(20).mean().iloc[-1])
    if vol_latest < vol_avg:
        return None

    ema9_15=float(ta.trend.EMAIndicator(close15,9).ema_indicator().iloc[-1])
    ema21_15=float(ta.trend.EMAIndicator(close15,21).ema_indicator().iloc[-1])
    rsi15=float(ta.momentum.RSIIndicator(close15,14).rsi().iloc[-1])

    last_close=float(close5.iloc[-1])
    prev_close=float(close5.iloc[-2])

    # guard against NaN values
    if math.isnan(last_close) or math.isnan(prev_close):
        return None

    bullish=(ema9_5>ema21_5 and rsi5>55 and ema9_15>ema21_15 and rsi15>55 and last_close>prev_close)
    bearish=(ema9_5<ema21_5 and rsi5<45 and ema9_15<ema21_15 and rsi15<45 and last_close<prev_close)

    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)

    fakeout=False
    try:
        if bullish and last_close<high_zone: fakeout=True
        if bearish and last_close>low_zone: fakeout=True
    except:
        fakeout=False

    # ✅ NEW: Integrate Liquidity Hunt Layer
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, fakeout

    if bullish: return "CE",df5,fakeout
    if bearish: return "PE",df5,fakeout
    return None

# --------- SYMBOL FORMAT FOR ALL INDICES ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    dt=datetime.strptime(expiry_str,"%d %b %Y")
    
    if index == "SENSEX":
        # SENSEX format: SENSEX + YY + MONTH_CODE + STRIKE + OPTION_TYPE
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    else:
        # Standard format for other indices
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

# --------- INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1])

    if vol_latest > vol_avg*1.5 and abs(last_close-prev_close)/prev_close>0.003:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg:
        return "PE"
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
    except:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        # Include both exchanges
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        # ensure 'oi' exists and convertible
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        if ce_sum>pe_sum*1.5: return "CE"
        if pe_sum>ce_sum*1.5: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- NEW INSTITUTIONAL CONFIRMATION LAYER ---------
def institutional_confirmation_layer(index, df5, base_signal):
    close = ensure_series(df5['Close'])
    high = ensure_series(df5['High'])
    low = ensure_series(df5['Low'])
    volume = ensure_series(df5['Volume'])
    atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    body_strength = abs(last_close - (high.iloc[-1]+low.iloc[-1])/2)

    # Liquidity trap: avoid if candle wicks through prev high/low zone
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
    if base_signal == 'CE' and last_close >= high_zone:
        return False
    if base_signal == 'PE' and last_close <= low_zone:
        return False

    # Weak momentum: skip if candle body small or low volume
    vol_avg = volume.rolling(20).mean().iloc[-1]
    if volume.iloc[-1] < vol_avg or body_strength < atr*0.25:
        return False

    # Volatility dead zone: skip low ATR periods
    if atr < (close.std() * 0.25):
        return False

    # Breadth check: ensure coherence across major index pairs
    if index == 'NIFTY':
        b_df = fetch_index_data('BANKNIFTY', '5m', '2d')
    elif index == 'BANKNIFTY':
        b_df = fetch_index_data('NIFTY', '5m', '2d')
    else:
        b_df = None

    if b_df is not None:
        b_close = ensure_series(b_df['Close'])
        b_ema9 = ta.trend.EMAIndicator(b_close, 9).ema_indicator().iloc[-1]
        b_ema21 = ta.trend.EMAIndicator(b_close, 21).ema_indicator().iloc[-1]
        if base_signal == 'PE' and b_ema9 > b_ema21:
            return False
        if base_signal == 'CE' and b_ema9 < b_ema21:
            return False

    return True

# --------- FLOW CONFIRMATION LAYER (UPDATED) ---------
def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    # ✅ NEW: Institutional Confirmation Add-On
    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- MONITOR WITH THREAD UPDATES ---------
def monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id):
    last_high = entry
    weakness_sent = False
    in_trade=False
    while True:
        # Auto stop at 3:30 PM IST
        if should_stop_trading():
            send_telegram(f"🛑 Market closed - Stopping monitoring for {symbol}", reply_to=thread_id)
            break
            
        price=fetch_option_price(symbol)
        if not price: time.sleep(10); continue
        price=round(price)
        if not in_trade:
            if price >= entry:
                send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                in_trade=True
                last_high=price
        else:
            if price > last_high:
                send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                last_high=price
            elif not weakness_sent and price < sl*1.05:
                send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                weakness_sent=True
            if price>=targets[0]:
                send_telegram(f"🌟 {symbol}: First Target {targets[0]} hit", reply_to=thread_id)
                break
            if price<=sl:
                send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                break
        time.sleep(10)

# --------- EXPIRY CONFIG FOR ALL INDICES ---------
EXPIRIES = {
    "NIFTY": "14 OCT 2025",
    "BANKNIFTY": "28 OCT 2025", 
    "SENSEX": "16 OCT 2025",
    "FINNIFTY": "28 OCT 2025",
    "MIDCPNIFTY": "28 OCT 2025",
    "EICHERMOT": "28 OCT 2025",
    "TRENT": "28 OCT 2025", 
    "RELIANCE": "28 OCT 2025"
}

# ACTIVE TRACKING FOR ALL INDICES
active_trades = {
    "NIFTY": None, "BANKNIFTY": None, "SENSEX": None,
    "FINNIFTY": None, "MIDCPNIFTY": None, "EICHERMOT": None,
    "TRENT": None, "RELIANCE": None
}

# --------- THREAD FUNCTION ---------
def trade_thread(index):
    global active_trades
    if active_trades[index]: return

    sig=analyze_index_signal(index)
    side=None; fakeout=False; df=None
    if sig: side, df, fakeout = sig

    df5=fetch_index_data(index,"5m","2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal=="BOTH":
        for s in ["CE","PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index,s,df,fakeout)
        return
    elif final_signal:
        if df is None: df=df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index,final_signal,df,fakeout)
    else:
        return

# --------- SEND SIGNAL ---------
def send_signal(index,side,df,fakeout):
    ltp=float(ensure_series(df["Close"]).iloc[-1])
    strike=round_strike(index,ltp)
    if strike is None:
        # cannot determine strike because LTP missing or invalid
        send_telegram(f"⚠️ {index}: could not determine strike (ltp missing). Signal skipped.")
        return
    symbol=get_option_symbol(index,EXPIRIES[index],strike,side)
    price=fetch_option_price(symbol)
    if not price: return
    high=ensure_series(df["High"])
    low=ensure_series(df["Low"])
    close=ensure_series(df["Close"])
    atr=float(ta.volatility.AverageTrueRange(high,low,close,14).average_true_range().iloc[-1])
    entry=round(price+5)
    sl=round(price-atr)
    targets=[round(price+atr*1.5),round(price+atr*2)]
    msg=(f" GIT🔊 {index} {side} VSSIGNAL CONFIRMED\n"
         f"🔹 Strike: {strike}\n"
         f"🟩 Buy Above ₹{entry}\n"
         f"🔵 SL: ₹{sl}\n"
         f"🌟 Targets: {targets[0]} / {targets[1]}\n"
         f"⚡ Fakeout: {'YES' if fakeout else 'NO'}")
    thread_id=send_telegram(msg)
    active_trades[index]={"symbol":symbol,"entry":entry,"sl":sl,"targets":targets,"thread":thread_id}
    monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id)
    active_trades[index]=None

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        print("🛑 Market closing time reached - stopping")
        send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
        exit(0)
        
    threads=[]
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    
    for index in all_indices:
        t=threading.Thread(target=trade_thread,args=(index,))
        t.start()
        threads.append(t)
    for t in threads: t.join()

# --------- START ---------
send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running in Parallel with Institutional Layers!")

while True:
    try:
        # Auto stop at 3:30 PM IST
        if should_stop_trading():
            send_telegram("🛑 Market closing time reached - Algorithm stopped automatically")
            break
            
        run_algo_parallel()
        time.sleep(30)
    except Exception as e:
        send_telegram(f"⚠️ Error in main loop: {e}")
        time.sleep(60)
