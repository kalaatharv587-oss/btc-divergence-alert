#!/usr/bin/env python3
"""
RSI divergence alerts for BTCUSD on Delta Exchange India -> Telegram.

- Timeframes: 5m and 1h
- Detects regular bullish and bearish RSI divergence
- Uses only the Python standard library (no pip install needed)
- Remembers what it already sent in state.json so you never get duplicates

Usage:
    python divergence_alert.py              # normal run (used by GitHub Actions)
    python divergence_alert.py --test       # send a test Telegram message
    python divergence_alert.py --dry-run    # check divergence, print only, send nothing

Environment variables (set as GitHub Secrets):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# ----------------------------- SETTINGS -----------------------------------
SYMBOL = "BTCUSD"
BASE_URL = "https://api.india.delta.exchange"
TIMEFRAMES = {"5m": 300, "1h": 3600}   # name -> seconds per candle
CANDLES_TO_FETCH = 300                 # history used for RSI + swing points

RSI_PERIOD = 14
PIVOT_LEFT = 5          # candles to the left that must be lower/higher than the pivot
PIVOT_RIGHT = 5         # candles to the right (confirmation delay = this many candles)
MIN_PIVOT_GAP = 5       # two pivots must be at least this many candles apart
MAX_PIVOT_GAP = 60      # ...and at most this many apart
MIN_RSI_DIFF = 2.0      # RSI must differ by at least this much between the pivots
BULL_ZONE_MAX = 45.0    # bullish: first pivot's RSI must be below this
BEAR_ZONE_MIN = 55.0    # bearish: first pivot's RSI must be above this
FRESH_WINDOW = 6        # only alert if the pivot was confirmed within this many candles

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
STATE_KEEP = 100        # how many sent-alert keys to remember
IST = timezone(timedelta(hours=5, minutes=30))
# ---------------------------------------------------------------------------


def http_get_json(url, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (rsi-divergence-alert)", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # network error, timeout, bad JSON
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} tries: {last_err}")


def fetch_candles(symbol, tf_name, tf_seconds, count):
    """Return a list of CLOSED candles, oldest first."""
    now = int(time.time())
    params = {
        "symbol": symbol,
        "resolution": tf_name,
        "start": now - (count + 2) * tf_seconds,
        "end": now,
    }
    url = f"{BASE_URL}/v2/history/candles?{urllib.parse.urlencode(params)}"
    data = http_get_json(url)
    if not data.get("success", True) or "result" not in data:
        raise RuntimeError(f"Unexpected API response: {str(data)[:300]}")
    candles = sorted(data["result"], key=lambda c: c["time"])  # API order is not guaranteed
    # Drop the candle still forming: it opened at c["time"] and closes at time + tf_seconds
    closed = [c for c in candles if c["time"] + tf_seconds <= now]
    return closed


def compute_rsi(closes, period=RSI_PERIOD):
    """Wilder's RSI (same as TradingView's RSI). Returns a list aligned with closes (None at start)."""
    rsi = [None] * len(closes)
    if len(closes) <= period:
        return rsi
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period

    def to_rsi(g, l):
        if l == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + g / l)

    rsi[period] = to_rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        rsi[i] = to_rsi(avg_gain, avg_loss)
    return rsi


def find_pivots(values, left, right, kind):
    """Indices of pivot highs (kind='high') or pivot lows (kind='low')."""
    pivots = []
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        v = values[i]
        if kind == "high" and v == max(window) and window.count(v) == 1:
            pivots.append(i)
        elif kind == "low" and v == min(window) and window.count(v) == 1:
            pivots.append(i)
    return pivots


def detect_divergences(candles):
    """Return a list of dicts describing divergences confirmed recently."""
    n = len(candles)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    rsi = compute_rsi(closes)
    found = []

    # Bullish: price lower low, RSI higher low
    lp = [i for i in find_pivots(lows, PIVOT_LEFT, PIVOT_RIGHT, "low") if rsi[i] is not None]
    if len(lp) >= 2:
        i1, i2 = lp[-2], lp[-1]
        gap = i2 - i1
        if (
            MIN_PIVOT_GAP <= gap <= MAX_PIVOT_GAP
            and lows[i2] < lows[i1]
            and rsi[i2] - rsi[i1] >= MIN_RSI_DIFF
            and rsi[i1] <= BULL_ZONE_MAX
            and (n - 1 - i2) <= PIVOT_RIGHT + FRESH_WINDOW
        ):
            found.append(dict(type="bullish", i1=i1, i2=i2, p1=lows[i1], p2=lows[i2], r1=rsi[i1], r2=rsi[i2]))

    # Bearish: price higher high, RSI lower high
    hp = [i for i in find_pivots(highs, PIVOT_LEFT, PIVOT_RIGHT, "high") if rsi[i] is not None]
    if len(hp) >= 2:
        i1, i2 = hp[-2], hp[-1]
        gap = i2 - i1
        if (
            MIN_PIVOT_GAP <= gap <= MAX_PIVOT_GAP
            and highs[i2] > highs[i1]
            and rsi[i1] - rsi[i2] >= MIN_RSI_DIFF
            and rsi[i1] >= BEAR_ZONE_MIN
            and (n - 1 - i2) <= PIVOT_RIGHT + FRESH_WINDOW
        ):
            found.append(dict(type="bearish", i1=i1, i2=i2, p1=highs[i1], p2=highs[i2], r1=rsi[i1], r2=rsi[i2]))

    return found, closes, rsi


def fmt_time(ts):
    return datetime.fromtimestamp(ts, IST).strftime("%d %b %H:%M IST")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sent": []}


def save_state(state):
    state["sent"] = state["sent"][-STATE_KEEP:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        # Do not print the URL (it contains the token)
        raise RuntimeError(f"Telegram error {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}") from None


def build_message(tf_name, candles, d, last_close, last_rsi):
    arrow = "🟢 BULLISH" if d["type"] == "bullish" else "🔴 BEARISH"
    kind = "lower low" if d["type"] == "bullish" else "higher high"
    rsi_kind = "higher low" if d["type"] == "bullish" else "lower high"
    return (
        f"{arrow} RSI divergence\n"
        f"{SYMBOL} | {tf_name}\n\n"
        f"Price: {d['p1']:,.1f} -> {d['p2']:,.1f} ({kind})\n"
        f"RSI: {d['r1']:.1f} -> {d['r2']:.1f} ({rsi_kind})\n"
        f"Swing points: {fmt_time(candles[d['i1']]['time'])} and {fmt_time(candles[d['i2']]['time'])}\n"
        f"Now: price {last_close:,.1f}, RSI {last_rsi:.1f}\n\n"
        f"Confirm on the chart and use a stop loss. This is a signal to look, not a trade."
    )


def main():
    args = set(sys.argv[1:])

    if "--test" in args:
        send_telegram("✅ Test message: your BTCUSD divergence alert bot is connected.")
        print("Test message sent.")
        return 0

    dry = "--dry-run" in args
    state = load_state()
    sent = set(state["sent"])
    errors = 0

    for tf_name, tf_seconds in TIMEFRAMES.items():
        try:
            candles = fetch_candles(SYMBOL, tf_name, tf_seconds, CANDLES_TO_FETCH)
            if len(candles) < RSI_PERIOD + 2 * (PIVOT_LEFT + PIVOT_RIGHT) + MAX_PIVOT_GAP:
                print(f"[{tf_name}] not enough candles ({len(candles)}), skipping")
                continue
            divs, closes, rsi = detect_divergences(candles)
            print(f"[{tf_name}] {len(candles)} candles, last close {closes[-1]:,.1f}, RSI {rsi[-1]:.1f}, divergences: {len(divs)}")
            for d in divs:
                key = f"{SYMBOL}-{tf_name}-{d['type']}-{candles[d['i2']]['time']}"
                if key in sent:
                    continue
                msg = build_message(tf_name, candles, d, closes[-1], rsi[-1])
                print(msg, "\n")
                if not dry:
                    send_telegram(msg)
                    sent.add(key)
                    state["sent"].append(key)
        except Exception as e:
            errors += 1
            print(f"[{tf_name}] ERROR: {e}")

    if not dry:
        save_state(state)
    return 1 if errors == len(TIMEFRAMES) else 0   # fail the run only if every timeframe failed


if __name__ == "__main__":
    sys.exit(main())
