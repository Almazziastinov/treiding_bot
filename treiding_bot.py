import requests
import time
import os
import csv
import numpy as np
from datetime import datetime

# ================== TELEGRAM ==================
TOKEN = "8377027532:AAGK7GvIyliq4oS-0AdZK9_6HTcQVfj4tWQ"
CHAT_ID = "842287010"
TG_URL = f"https://api.telegram.org/bot{TOKEN}"

def tg_send(text):
    requests.post(
        f"{TG_URL}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=10
    )

# ================== CONFIG ==================
CONFIG = {
    "TIMEFRAME": "15m",
    "RISK": 10,
    "TP_FIXED": 4,   # +4%
    "ENABLED": True
}

SYMBOLS = ["BTC-USDT","ETH-USDT","SOL-USDT","SUI-USDT","TIA-USDT"]
CHECK_INTERVAL = 30
CSV_FILE = "trades.csv"

# ================== CSV ==================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["date","symbol","side","entry","sl","tp1","tp2","result"]
        )

def log_trade(row):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

def update_trade(symbol, result):
    rows = []
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.reader(f))

    for i in range(len(rows)-1, 0, -1):
        if rows[i][1] == symbol and rows[i][-1] == "OPEN":
            rows[i][-1] = result
            break

    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

# ================== TELEGRAM COMMANDS ==================
OFFSET = 0

def get_updates():
    global OFFSET
    r = requests.get(
        f"{TG_URL}/getUpdates",
        params={"offset": OFFSET, "timeout": 10}
    ).json()

    for u in r.get("result", []):
        OFFSET = u["update_id"] + 1
        yield u

def handle_commands():
    for u in get_updates():
        msg = u.get("message", {})
        if str(msg.get("chat", {}).get("id")) != CHAT_ID:
            continue

        text = msg.get("text", "")

        if text.startswith("/risk"):
            CONFIG["RISK"] = int(text.split()[1])
            tg_send(f"✅ Risk: {CONFIG['RISK']}%")

        elif text.startswith("/tf"):
            CONFIG["TIMEFRAME"] = text.split()[1]
            tg_send(f"⏱ TF: {CONFIG['TIMEFRAME']}")

        elif text.startswith("/tp"):
            CONFIG["TP_FIXED"] = int(text.split()[1])
            tg_send(f"🎯 TP2: {CONFIG['TP_FIXED']}%")

        elif text == "/on":
            CONFIG["ENABLED"] = True
            tg_send("🟢 Signals ON")

        elif text == "/off":
            CONFIG["ENABLED"] = False
            tg_send("🔴 Signals OFF")

        elif text == "/status":
            tg_send(str(CONFIG))

        elif text == "/stats":
            with open(CSV_FILE) as f:
                total = sum(1 for _ in f) - 1
            tg_send(f"📊 Trades logged: {total}")

# ================== MARKET ==================
def get_klines(symbol, limit=300):
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
    r = requests.get(url, params={
        "symbol": symbol,
        "interval": CONFIG["TIMEFRAME"],
        "limit": limit
    }, timeout=10).json()
    return r["data"] if r.get("code") == 0 else []

def ema(data, period):
    return np.mean(data[-period:])

def atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        ))
    return np.mean(trs[-period:])

def quality_filter(closes, volumes, highs, lows):
    atr_val = atr(highs, lows, closes)
    avg_range = np.mean([h-l for h,l in zip(highs[-20:],lows[-20:])])
    impulse = abs(closes[-1] - closes[-2])
    avg_impulse = np.mean([abs(closes[i]-closes[i-1]) for i in range(-10,-1)])

    if atr_val < avg_range * 0.3:
        return False
    if impulse < avg_impulse * 1.2:
        return False
    if volumes[-1] < np.mean(volumes[-20:]) * 1.3:
        return False
    return True

# ================== ANALYZE ==================
def analyze(symbol):
    k = get_klines(symbol)
    if len(k) < 200:
        return None

def main():
    while True:
        try:
            handle_commands()
            if CONFIG["ENABLED"]:
                for symbol in SYMBOLS:
                    signal = analyze(symbol)
                    if signal:
                        tg_send(signal)
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()