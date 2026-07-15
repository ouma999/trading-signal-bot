"""
Trading Signal Telegram Alert Bot
===================================
Same free data sources as the dashboard version (yfinance + Google News RSS +
VADER sentiment) but instead of writing an HTML file, it pushes an alert to
your phone via Telegram whenever a strong signal appears -- similar to the
forex bot you built earlier, but using free data instead of paid API keys.

HONEST DISCLAIMER (same as before, worth repeating)
-----------------------------------------------------
This does NOT predict the market. The "confidence" score is a measure of how
many of your chosen signals currently agree with each other -- not a
probability of the trade working out. No system can promise that. You still
make every entry/exit decision yourself.

SETUP -- ONE TIME
------------------
1. Install dependencies:
     pip3 install yfinance feedparser vaderSentiment pandas requests schedule

2. Create a Telegram bot (2 minutes, free):
   a. Open Telegram, search for "BotFather", start a chat
   b. Send: /newbot
   c. Follow the prompts (choose a name, choose a username ending in "bot")
   d. BotFather gives you a token like: 123456789:AAExampleTokenHere
   e. Paste it into BOT_TOKEN below

3. Get your chat ID:
   a. Send any message to your new bot (search its username, say "hi")
   b. In your browser, visit:
        https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
      (replace <YOUR_TOKEN> with your real token)
   c. Look for "chat":{"id": 123456789 ...} in the response
   d. Paste that number into CHAT_ID below

4. Run it:
     python3 telegram_alert_bot.py

It sends a test message immediately, then checks every CHECK_INTERVAL_MINUTES
and only messages you again when a NEW strong signal appears (won't spam you
with the same signal repeatedly).
"""

import os
import time
import datetime as dt
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Reads from environment variables first (used on Railway/cloud deployment).
# Falls back to the hardcoded strings below for local testing only --
# if you deploy, set these as Variables in Railway instead of editing here.
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8857807265:AAHyh8ODpp7P3S8eCxrhc7gRgfxXeV8cR2g")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7887326351")

# FRED (Federal Reserve Economic Data) -- free API key, get one at:
# https://fred.stlouisfed.org/docs/api/api_key.html (instant, no cost)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "45a2afb8f3bf7985902f0aba422db078")

# Confirmed 2026 FOMC meeting dates (verified against federalreserve.gov, July 2026).
# The rate decision is announced on the SECOND day of each meeting.
FOMC_MEETING_DATES_2026 = [
    dt.date(2026, 1, 28),
    dt.date(2026, 3, 18),
    dt.date(2026, 4, 29),
    dt.date(2026, 6, 17),
    dt.date(2026, 7, 29),
    dt.date(2026, 9, 16),
    dt.date(2026, 10, 28),
    dt.date(2026, 12, 9),
]
EVENT_WARNING_DAYS = 3       # flag FOMC meetings within this many days
EARNINGS_WARNING_DAYS = 7    # flag earnings within this many days

# Which direction each instrument tends to move when US rates FALL.
# +1 = bullish when rates fall (bearish when rates rise)
# -1 = bearish when rates fall (bullish when rates rise)
# Simplified real-world logic:
#   Falling rates -> weaker USD -> bullish gold & oil (priced in USD),
#                    bullish growth stocks like NVDA (cheaper future cash flows),
#                    bearish USD/CAD, bearish bank stocks (lower net interest margin)
MACRO_RATE_SENSITIVITY = {
    "NVDA":   1,
    "GS":    -1,
    "JPM":   -1,
    "XAUUSD": 1,
    "WTI":    1,
    "USDCAD":-1,
}

# is_stock=True  -> only checked/alerted during NYSE hours (9:30am-4:00pm ET, Mon-Fri)
# is_stock=False -> trades ~24/5 (forex, gold, oil), checked continuously
WATCHLIST = {
    "NVDA":   {"name": "Nvidia",         "yf_ticker": "NVDA",    "news_query": "Nvidia stock",     "is_stock": True},
    "GS":     {"name": "Goldman Sachs",  "yf_ticker": "GS",      "news_query": "Goldman Sachs stock", "is_stock": True},
    "JPM":    {"name": "JPMorgan",       "yf_ticker": "JPM",     "news_query": "JPMorgan stock",   "is_stock": True},
    "XAUUSD": {"name": "Gold (XAU/USD)", "yf_ticker": "GC=F",    "news_query": "gold price",       "is_stock": False},
    "WTI":    {"name": "WTI Crude Oil",  "yf_ticker": "CL=F",    "news_query": "WTI oil price",    "is_stock": False},
    "USDCAD": {"name": "USD/CAD",        "yf_ticker": "USDCAD=X","news_query": "USD CAD forex",    "is_stock": False},
}

CHECK_INTERVAL_MINUTES = 30     # how often to re-check the watchlist
MIN_ABS_SCORE_TO_ALERT = 2      # only alert on ALIGNED signals (score +-2 or more)
ATR_STOP_MULTIPLE = 1.5         # stop loss = price -+ (ATR * this)
ATR_TARGET_MULTIPLE = 2.5       # take profit = price -+ (ATR * this) -> ~1.6:1 reward:risk

# NYSE regular trading hours, in US Eastern time (handles EST/EDT automatically)
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
PRE_MARKET_HEADSUP_MINUTES = 30   # send a heads-up this many minutes before open

RSI_PERIOD = 14
SMA_FAST = 20
SMA_SLOW = 50
ATR_PERIOD = 14
SWING_LOOKBACK = 20
NEWS_MAX_ITEMS = 8

# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS (same logic as the dashboard version)
# ---------------------------------------------------------------------------

def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def get_technicals(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < SMA_SLOW:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["SMA_fast"] = df["Close"].rolling(SMA_FAST).mean()
    df["SMA_slow"] = df["Close"].rolling(SMA_SLOW).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["ATR"] = compute_atr(df)
    macd_line, signal_line, hist = compute_macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist
    df["Vol_avg20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent = df.tail(SWING_LOOKBACK)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())

    trend = "bullish" if float(last["SMA_fast"]) > float(last["SMA_slow"]) else "bearish"
    rsi_val = float(last["RSI"])
    if rsi_val >= 70:
        rsi_state = "overbought"
    elif rsi_val <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"

    macd_bullish = float(last["MACD"]) > float(last["MACD_signal"])
    macd_prev_bullish = float(prev["MACD"]) > float(prev["MACD_signal"])
    macd_just_crossed = macd_bullish != macd_prev_bullish

    current_volume = float(last["Volume"])
    avg_volume = float(last["Vol_avg20"]) if pd.notna(last["Vol_avg20"]) else current_volume
    if avg_volume <= 0:
        # Forex pairs / some futures report no reliable volume via yfinance —
        # treat as "no data" rather than penalizing the signal unfairly.
        volume_ratio = None
        volume_confirmed = None
    else:
        volume_ratio = current_volume / avg_volume
        volume_confirmed = volume_ratio >= 1.2

    return {
        "last_price": float(last["Close"]),
        "trend": trend,
        "rsi": rsi_val,
        "rsi_state": rsi_state,
        "atr": float(last["ATR"]),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "macd_bullish": macd_bullish,
        "macd_just_crossed": macd_just_crossed,
        "volume_ratio": volume_ratio,
        "volume_confirmed": volume_confirmed,
    }


# ---------------------------------------------------------------------------
# NEWS SENTIMENT
# ---------------------------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(query, max_items=NEWS_MAX_ITEMS):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    entries = feed.entries[:max_items]

    if not entries:
        return {"avg_sentiment": 0.0, "count": 0}

    scores = [analyzer.polarity_scores(e.get("title", ""))["compound"] for e in entries]
    return {"avg_sentiment": sum(scores) / len(scores), "count": len(scores)}


# ---------------------------------------------------------------------------
# MACRO / FUNDAMENTAL LAYER (FRED interest rates + FOMC/earnings event risk)
# ---------------------------------------------------------------------------

_fred_cache = {}  # simple in-memory cache so we don't hammer the API every check

def get_fed_funds_rate_trend():
    """Returns 'rising', 'falling', 'flat', or None (if unavailable)."""
    if "fed_trend" in _fred_cache:
        return _fred_cache["fed_trend"]

    if not FRED_API_KEY or FRED_API_KEY == "PASTE_YOUR_FRED_KEY_HERE":
        return None

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "FEDFUNDS",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 3,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json()["observations"]
        values = [float(o["value"]) for o in obs if o["value"] != "."]

        if len(values) < 2:
            trend = None
        elif values[0] > values[1]:
            trend = "rising"
        elif values[0] < values[1]:
            trend = "falling"
        else:
            trend = "flat"

        _fred_cache["fed_trend"] = trend
        return trend
    except Exception as e:
        print(f"  FRED fetch failed: {e}")
        return None


def get_macro_signal(key):
    """Returns (score_contribution, reason_text) based on Fed rate trend."""
    trend = get_fed_funds_rate_trend()
    sensitivity = MACRO_RATE_SENSITIVITY.get(key, 0)

    if trend is None or sensitivity == 0:
        return 0, None

    if trend == "rising":
        contribution = -sensitivity
    elif trend == "falling":
        contribution = sensitivity
    else:
        return 0, "Fed funds rate flat -- no macro bias"

    direction = "bullish" if contribution > 0 else "bearish"
    return contribution, f"macro: Fed rate {trend} -- {direction} bias for this instrument"


def get_event_risk_warnings(key, meta):
    """Returns a list of warning strings for upcoming high-impact events."""
    warnings_list = []
    today = dt.date.today()

    # FOMC meeting proximity (affects all instruments, especially gold/financials)
    for meeting_date in FOMC_MEETING_DATES_2026:
        days_away = (meeting_date - today).days
        if 0 <= days_away <= EVENT_WARNING_DAYS:
            warnings_list.append(f"FOMC meeting in {days_away} day(s) ({meeting_date}) -- expect volatility")
            break

    # Earnings proximity (stocks only)
    if meta.get("is_stock"):
        try:
            ticker_obj = yf.Ticker(meta["yf_ticker"])
            earnings_dates = ticker_obj.get_earnings_dates(limit=4)
            if earnings_dates is not None and not earnings_dates.empty:
                for idx in earnings_dates.index:
                    edate = idx.date() if hasattr(idx, "date") else idx
                    days_away = (edate - today).days
                    if 0 <= days_away <= EARNINGS_WARNING_DAYS:
                        warnings_list.append(f"Earnings in {days_away} day(s) ({edate}) -- expect a gap risk")
                        break
        except Exception as e:
            print(f"  Earnings calendar fetch failed for {key}: {e}")

    return warnings_list


# ---------------------------------------------------------------------------
# COMPOSITE SIGNAL + STOP LOSS / TAKE PROFIT
# ---------------------------------------------------------------------------

def build_signal(tech, sentiment, macro_key=None, event_warnings=None):
    if tech is None:
        return None

    score = 0
    reasons = []

    if tech["trend"] == "bullish":
        score += 1
        reasons.append("trend bullish")
    else:
        score -= 1
        reasons.append("trend bearish")

    if tech["rsi_state"] == "oversold":
        score += 1
        reasons.append(f"RSI oversold ({tech['rsi']:.1f})")
    elif tech["rsi_state"] == "overbought":
        score -= 1
        reasons.append(f"RSI overbought ({tech['rsi']:.1f})")

    if tech["macd_bullish"]:
        score += 1
        reasons.append("MACD bullish" + (" (just crossed)" if tech["macd_just_crossed"] else ""))
    else:
        score -= 1
        reasons.append("MACD bearish" + (" (just crossed)" if tech["macd_just_crossed"] else ""))

    if sentiment["avg_sentiment"] > 0.15:
        score += 1
        reasons.append(f"news sentiment positive ({sentiment['avg_sentiment']:.2f})")
    elif sentiment["avg_sentiment"] < -0.15:
        score -= 1
        reasons.append(f"news sentiment negative ({sentiment['avg_sentiment']:.2f})")

    # Macro layer: Fed rate trend, mapped per-instrument (see MACRO_RATE_SENSITIVITY)
    macro_contribution, macro_reason = (0, None)
    if macro_key is not None:
        macro_contribution, macro_reason = get_macro_signal(macro_key)
        if macro_contribution != 0:
            score += macro_contribution
        if macro_reason:
            reasons.append(macro_reason)

    direction = "BUY" if score > 0 else ("SELL" if score < 0 else "NEUTRAL")

    # confidence = how many of the 5 signal components agree
    # (trend, RSI, MACD, sentiment, macro) -- macro only counts if available
    max_components = 5 if macro_key is not None else 4
    base_confidence = min(abs(score), max_components) / max_components * 100

    # volume acts as a modifier, not a scored component -- confirms conviction
    # behind a move rather than adding a directional vote of its own
    if tech["volume_confirmed"] is None:
        confidence = round(base_confidence)
        reasons.append("volume data unavailable for this instrument")
    elif tech["volume_confirmed"]:
        confidence = round(min(base_confidence * 1.1, 100))
        reasons.append(f"volume confirms move ({tech['volume_ratio']:.1f}x avg)")
    else:
        confidence = round(base_confidence * 0.75)
        reasons.append(f"volume below average ({tech['volume_ratio']:.1f}x avg) — weaker conviction")

    # Event-risk warnings (FOMC, earnings) -- informational, not scored,
    # since a scheduled event can invalidate technical/sentiment signals via a gap
    if event_warnings:
        reasons.extend(f"⚠️ {w}" for w in event_warnings)

    price = tech["last_price"]
    atr = tech["atr"]
    if direction == "BUY":
        stop_loss = price - (atr * ATR_STOP_MULTIPLE)
        take_profit = price + (atr * ATR_TARGET_MULTIPLE)
    elif direction == "SELL":
        stop_loss = price + (atr * ATR_STOP_MULTIPLE)
        take_profit = price - (atr * ATR_TARGET_MULTIPLE)
    else:
        stop_loss = take_profit = None

    return {
        "score": score,
        "direction": direction,
        "confidence": confidence,
        "reasons": reasons,
        "price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


def format_alert(key, meta, sig):
    arrow = "🟢 BUY" if sig["direction"] == "BUY" else ("🔴 SELL" if sig["direction"] == "SELL" else "⚪ NEUTRAL")
    reasons_txt = "\n".join(f"  • {r}" for r in sig["reasons"])
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"🚨 <b>SIGNAL ALERT — {meta['name']} ({key})</b>",
        f"Time: {now}",
        f"Direction: {arrow}",
        f"Confidence: {sig['confidence']}%",
        f"Price: {sig['price']:.4f}",
    ]
    if sig["stop_loss"] is not None:
        lines.append(f"Stop Loss: {sig['stop_loss']:.4f}")
        lines.append(f"Take Profit: {sig['take_profit']:.4f}")
    lines.append("Why:")
    lines.append(reasons_txt)
    lines.append("")
    lines.append("⚠️ Not financial advice. Confidence = how many signals agree, not a win probability. Confirm before trading.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MARKET HOURS (NYSE, US Eastern time)
# ---------------------------------------------------------------------------

def get_market_status():
    """Returns one of: 'weekend', 'pre_market_headsup', 'open', 'closed'."""
    now_et = dt.datetime.now(MARKET_TZ)

    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return "weekend", now_et

    now_t = now_et.time()
    headsup_start = (dt.datetime.combine(now_et.date(), MARKET_OPEN)
                      - dt.timedelta(minutes=PRE_MARKET_HEADSUP_MINUTES)).time()

    if headsup_start <= now_t < MARKET_OPEN:
        return "pre_market_headsup", now_et
    elif MARKET_OPEN <= now_t < MARKET_CLOSE:
        return "open", now_et
    else:
        return "closed", now_et


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

last_alerted = {}          # key -> (direction, score) to avoid repeat alerts
last_headsup_date = None   # avoid sending the pre-market heads-up more than once/day

def send_premarket_headsup():
    stock_names = [meta["name"] for meta in WATCHLIST.values() if meta["is_stock"]]
    if not stock_names:
        return
    msg = (
        f"⏰ <b>Market opens in {PRE_MARKET_HEADSUP_MINUTES} minutes</b>\n"
        f"Watching: {', '.join(stock_names)}\n"
        f"Signal checks for these will resume once the market opens."
    )
    send_telegram_message(msg)
    print("  -> Pre-market heads-up sent")


def check_and_alert():
    global last_headsup_date
    status, now_et = get_market_status()
    print(f"\n[{now_et.strftime('%H:%M:%S %Z')}] Market status: {status}")

    if status == "pre_market_headsup" and last_headsup_date != now_et.date():
        send_premarket_headsup()
        last_headsup_date = now_et.date()

    for key, meta in WATCHLIST.items():
        # Skip stock instruments outside market hours (they'd just show stale data)
        if meta["is_stock"] and status != "open":
            print(f"  {key}: market closed, skipping")
            continue

        try:
            tech = get_technicals(meta["yf_ticker"])
            sentiment = get_news_sentiment(meta["news_query"])
            event_warnings = get_event_risk_warnings(key, meta)
            sig = build_signal(tech, sentiment, macro_key=key, event_warnings=event_warnings)

            if sig is None:
                print(f"  {key}: no data")
                continue

            print(f"  {key}: {sig['direction']} score={sig['score']} confidence={sig['confidence']}%")

            if abs(sig["score"]) >= MIN_ABS_SCORE_TO_ALERT:
                signature = (sig["direction"], sig["score"])
                if last_alerted.get(key) != signature:
                    last_alerted[key] = signature
                    message = format_alert(key, meta, sig)
                    if send_telegram_message(message):
                        print(f"  -> Alert sent for {key}")
                else:
                    print(f"  -> Same signal as last check, skipping duplicate alert")

        except Exception as e:
            print(f"  {key}: error - {e}")


def main():
    print("Sending test message to confirm Telegram is connected...")
    ok = send_telegram_message("✅ Trading signal bot connected. You'll get alerts here when a strong signal appears.")
    if not ok:
        print("Test message failed -- check your BOT_TOKEN and CHAT_ID before continuing.")
        return

    check_and_alert()

    print(f"\nRunning continuously, checking every {CHECK_INTERVAL_MINUTES} minutes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
            check_and_alert()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()