import os
import logging
import csv
import numpy as np
import requests
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, JobQueue

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ================== CONFIG & GLOBALS ==================
ALLOWED_CHAT_IDS = ["842287010", "635124229"]
CONFIG = {
    "TIMEFRAME": "15m",
    "RISK": 10,
    "TP_FIXED": 4,   # +4%
    "ENABLED": True
}
SYMBOLS = []
SYMBOLS_FILE = "bingx_futures_symbols.txt"
try:
    with open(SYMBOLS_FILE, 'r') as f:
        SYMBOLS = [line.strip() for line in f if line.strip()]
    logging.info(f"Successfully loaded {len(SYMBOLS)} symbols from {SYMBOLS_FILE}.")
except FileNotFoundError:
    logging.error(f"Symbols file not found: {SYMBOLS_FILE}. The bot will not analyze any symbols.")
CSV_FILE = "trades.csv"

# ================== CSV ==================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["date", "symbol", "side", "entry", "sl", "tp1", "tp2", "result"]
        )

def log_trade(row):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

def update_trade(symbol, result):
    rows = []
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.reader(f))
    for i in range(len(rows) - 1, 0, -1):
        if rows[i][1] == symbol and rows[i][-1] == "OPEN":
            rows[i][-1] = result
            break
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)

# ================== MARKET & INDICATORS ==================
def get_klines(symbol, limit=300):
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
    try:
        r = requests.get(url, params={
            "symbol": symbol,
            "interval": CONFIG["TIMEFRAME"],
            "limit": limit
        }, timeout=10).json()
        return r["data"] if r.get("code") == 0 else []
    except requests.RequestException as e:
        logging.error(f"Error fetching klines for {symbol}: {e}")
        return []

def ema(data, period):
    if len(data) < period: return None
    return np.mean(data[-period:])

def atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    if not trs: return 0
    return np.mean(trs[-period:])

def quality_filter(closes, volumes, highs, lows):
    try:
        if len(closes) < 21 or len(volumes) < 21: return False
        atr_val = atr(highs, lows, closes)
        avg_range = np.mean([h - l for h, l in zip(highs[-20:], lows[-20:])])
        impulse = abs(closes[-1] - closes[-2])
        avg_impulse = np.mean([abs(closes[i] - closes[i-1]) for i in range(-10, -1)])
        if atr_val < avg_range * 0.3: return False
        if impulse < avg_impulse * 1.2: return False
        if volumes[-1] < np.mean(volumes[-20:]) * 1.3: return False
        return True
    except (IndexError, ValueError) as e:
        logging.error(f"Error in quality_filter: {e}")
        return False

def find_fvg(highs, lows):
    for i in range(-3, -15, -1):
        if highs[i] < lows[i-2]: return ('BULLISH', highs[i], lows[i-2])
        if lows[i] > highs[i-2]: return ('BEARISH', highs[i-2], lows[i])
    return None

def find_confirmed_ob(closes, highs, lows, volumes, lookback=15, confirmation_multiplier=1.5):
    if len(closes) < lookback + 5: return None
    avg_volume = np.mean(volumes[-lookback:])
    # Search for a Bullish OB (last red candle before an up-move)
    for i in range(-3, -lookback, -1):
        if closes[i] < closes[i-1] and closes[i+1] > closes[i]:
            for j in range(i + 1, min(i + 4, 0)):
                if volumes[j] > avg_volume * confirmation_multiplier:
                    return (lows[i], highs[i])
    # Search for a Bearish OB (last green candle before a down-move)
    for i in range(-3, -lookback, -1):
        if closes[i] > closes[i-1] and closes[i+1] < closes[i]:
            for j in range(i + 1, min(i + 4, 0)):
                if volumes[j] > avg_volume * confirmation_multiplier:
                    return (lows[i], highs[i])
    return None

# ================== STRATEGY & ANALYSIS ==================
def analyze(symbol):
    k = get_klines(symbol)
    if len(k) < 250: return None

    closes = [float(x['close']) for x in k]
    highs, lows, vols = [float(x['high']) for x in k], [float(x['low']) for x in k], [float(x['volume']) for x in k]
    ema50, ema200 = ema(closes, 50), ema(closes, 200)

    if ema50 is None or ema200 is None or not quality_filter(closes, vols, highs, lows):
        return None

    signal_data = None

    # --- Setup 1: Retest ---
    resistance_level = max(highs[-20:-3])
    if (ema50 > ema200 and closes[-3] > resistance_level and lows[-2] <= resistance_level and closes[-2] > resistance_level and closes[-1] > closes[-2]):
        price = closes[-1]
        sl = min(lows[-5:]) * 0.99
        tp1, tp2 = price + (price - sl) * 2, price * (1 + CONFIG["TP_FIXED"]/100)
        signal_data = ("LONG", price, sl, tp1, tp2, (resistance_level, price))

    support_level = min(lows[-20:-3])
    if not signal_data and (ema50 < ema200 and closes[-3] < support_level and highs[-2] >= support_level and closes[-2] < support_level and closes[-1] < closes[-2]):
        price = closes[-1]
        sl = max(highs[-5:]) * 1.01
        tp1, tp2 = price - (sl - price) * 2, price * (1 - CONFIG["TP_FIXED"]/100)
        signal_data = ("SHORT", price, sl, tp1, tp2, (price, support_level))

    # --- Setup 2: EMA/FVG Collision ---
    if not signal_data:
        fvg_info = find_fvg(highs, lows)
        if fvg_info:
            fvg_type, fvg_low, fvg_high = fvg_info
            price = closes[-1]
            if fvg_low < ema200 < fvg_high and abs(price - ema200) / price < 0.001:
                entry_range = (fvg_low, fvg_high)
                if fvg_type == 'BULLISH' and price > ema200:
                    sl = fvg_low * 0.99
                    tp1, tp2 = price + (price - sl) * 2, price * (1 + CONFIG["TP_FIXED"]/100)
                    signal_data = ("LONG", price, sl, tp1, tp2, entry_range)
                elif fvg_type == 'BEARISH' and price < ema200:
                    sl = fvg_high * 1.01
                    tp1, tp2 = price - (sl - price) * 2, price * (1 - CONFIG["TP_FIXED"]/100)
                    signal_data = ("SHORT", price, sl, tp1, tp2, entry_range)

    # --- Final Check: OB Confirmation (Enhancer) ---
    if signal_data:
        ob_confirmed = False
        ob_info = find_confirmed_ob(closes, highs, lows, vols)
        if ob_info:
            ob_low, ob_high = ob_info
            entry_price = signal_data[1]
            if ob_low <= entry_price <= ob_high:
                ob_confirmed = True
        return signal_data + (ob_confirmed,)

    return None

# ================== BACKGROUND JOB ==================
async def run_analysis(context: ContextTypes.DEFAULT_TYPE):
    if not CONFIG["ENABLED"]: return
    logging.info("Running periodic analysis for symbols...")
    for symbol in SYMBOLS:
        try:
            signal_data = analyze(symbol)
            if signal_data:
                side, price, sl, tp1, tp2, entry_range, ob_confirmed = signal_data
                
                signal_message = (
                    f"<b>New Signal for {symbol}</b>\n\n"
                    f"<b>Side:</b> {side}\n"
                    f"<b>Entry Range:</b> {entry_range[0]:.4f} - {entry_range[1]:.4f}\n"
                    f"<b>Entry Price:</b> {price:.4f}\n"
                    f"<b>Stop Loss:</b> {sl:.4f}\n"
                    f"<b>Take Profit 1:</b> {tp1:.4f}\n"
                    f"<b>Take Profit 2:</b> {tp2:.4f}"
                )
                if ob_confirmed:
                    signal_message += "\n<b>Confirmation:</b> ✅ OB Confirmed"
                
                log_trade([datetime.now().strftime('%Y-%m-%d %H:%M'), symbol, side, price, sl, tp1, tp2, "OPEN"])
                for chat_id in context.job.data.get("chat_ids", []):
                    await context.bot.send_message(chat_id=chat_id, text=signal_message, parse_mode='HTML')
        except Exception as e:
            logging.error(f"Error during analysis of {symbol}: {e}")

# ================== TELEGRAM COMMAND HANDLERS ==================
def restricted(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if str(update.effective_chat.id) not in ALLOWED_CHAT_IDS:
            logging.warning(f"Unauthorized access denied for chat ID: {update.effective_chat.id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your signal bot. I am ready to work.")

@restricted
async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        new_risk = int(context.args[0])
        CONFIG["RISK"] = new_risk
        await update.message.reply_text(f"✅ Risk set to: {CONFIG['RISK']}%")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /risk <percentage>")

@restricted
async def timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        new_tf = context.args[0]
        CONFIG["TIMEFRAME"] = new_tf
        await update.message.reply_text(f"⏱ Timeframe set to: {CONFIG['TIMEFRAME']}")
    except IndexError:
        await update.message.reply_text("Usage: /tf <timeframe>")

@restricted
async def tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        new_tp = int(context.args[0])
        CONFIG["TP_FIXED"] = new_tp
        await update.message.reply_text(f"🎯 TP2 set to: {CONFIG['TP_FIXED']}%")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /tp <percentage>")

@restricted
async def on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["ENABLED"] = True
    await update.message.reply_text("🟢 Signals ON")

@restricted
async def off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CONFIG["ENABLED"] = False
    await update.message.reply_text("🔴 Signals OFF")

@restricted
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_text = "<pre>" + "\n".join([f"{k}: {v}" for k, v in CONFIG.items()]) + "</pre>"
    await update.message.reply_html(status_text)

@restricted
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open(CSV_FILE) as f:
            total = sum(1 for _ in f) - 1
        await update.message.reply_text(f"📊 Trades logged: {total}")
    except FileNotFoundError:
        await update.message.reply_text("Trade log file not found.")

# ================== MAIN ==================
def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logging.error("Error: BOT_TOKEN environment variable not set.")
        return

    application = ApplicationBuilder().token(token).build()
    job_queue = application.job_queue
    job_queue.run_repeating(run_analysis, interval=60, first=10, data={"chat_ids": ALLOWED_CHAT_IDS})

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('risk', risk))
    application.add_handler(CommandHandler('tf', timeframe))
    application.add_handler(CommandHandler('tp', tp))
    application.add_handler(CommandHandler('on', on))
    application.add_handler(CommandHandler('off', off))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('stats', stats))

    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()