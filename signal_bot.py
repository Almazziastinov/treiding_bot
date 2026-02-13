import os
import logging
import csv
import numpy as np
import requests
from datetime import datetime, timedelta
import asyncio

# Импортируем наш новый сканер
import market_scanner

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
    "ENABLED": True,
    "TIMEFRAME": "15m", # Таймфрейм по умолчанию
    "MAX_OPEN_TRADES": 5, # Максимальное количество одновременных сделок
    "ATR_SL_MULTIPLIER": 1.5, # Множитель ATR для стоп-лосса
    "RR_TP1": 1.5, # Risk/Reward для TP1
    "RR_TP2": 3.0,  # Risk/Reward для TP2
    "COOLDOWN_PERIOD_MINUTES": 60, # Период охлаждения для символа в минутах
}

# Глобальный список отслеживаемых монет, обновляется сканером
WATCHLIST = []
# Словарь для отслеживания времени последнего сигнала по монете
SIGNAL_COOLDOWN = {}

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

def get_open_trades_count():
    """Возвращает количество открытых сделок из CSV."""
    try:
        with open(CSV_FILE, 'r', newline='') as f:
            trades = list(csv.reader(f))
        if not trades or len(trades) <= 1:
            return 0
        open_trades = [row for row in trades[1:] if row[7] == "OPEN"]
        return len(open_trades)
    except (FileNotFoundError, StopIteration):
        return 0

def update_trade(symbol, result):
    rows = []
    try:
        with open(CSV_FILE, 'r', newline='') as f:
            rows = list(csv.reader(f))

        for i in range(len(rows) - 1, 0, -1):
            if rows[i][1] == symbol and rows[i][7] == "OPEN":
                rows[i][7] = result
                break

        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
    except FileNotFoundError:
        logging.error("trades.csv not found while trying to update trade.")

# ================== MARKET & INDICATORS ==================
def get_klines(symbol, limit=300):
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": CONFIG["TIMEFRAME"], "limit": limit}, timeout=10).json()
        return r.get("data", []) if r.get("code") == 0 else []
    except requests.RequestException as e:
        logging.error(f"Error fetching klines for {symbol}: {e}")
        return []

def get_current_price(symbol):
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/price"
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=5).json()
        if r.get("code") == 0 and 'data' in r and 'price' in r['data']:
            return float(r['data']['price'])
    except (requests.RequestException, KeyError, ValueError) as e:
        logging.error(f"Could not fetch current price for {symbol}: {e}")
    return None

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
    for i in range(-3, -lookback, -1):
        if closes[i] < closes[i-1] and closes[i+1] > closes[i]:
            for j in range(i + 1, min(i + 4, 0)):
                if volumes[j] > avg_volume * confirmation_multiplier: return (lows[i], highs[i])
    for i in range(-3, -lookback, -1):
        if closes[i] > closes[i-1] and closes[i+1] < closes[i]:
            for j in range(i + 1, min(i + 4, 0)):
                if volumes[j] > avg_volume * confirmation_multiplier: return (lows[i], highs[i])
    return None

# ================== STRATEGY & ANALYSIS ==================
def analyze(symbol):
    # --- Новая логика: Отключить SHORT сделки ---
    # Мы можем сделать это более гибким в будущем, но пока отключаем полностью
    ALLOW_SHORTS = False

    k = get_klines(symbol)
    if len(k) < 250: return None
    closes = [float(x['close']) for x in k]
    highs, lows, vols = [float(x['high']) for x in k], [float(x['low']) for x in k], [float(x['volume']) for x in k]
    
    ema50, ema200 = ema(closes, 50), ema(closes, 200)
    atr_val = atr(highs, lows, closes, period=14)

    if ema50 is None or ema200 is None or atr_val == 0 or not quality_filter(closes, vols, highs, lows):
        return None
        
    signal_data = None
    price = closes[-1]
    sl_multiplier = CONFIG["ATR_SL_MULTIPLIER"]
    tp1_rr, tp2_rr = CONFIG["RR_TP1"], CONFIG["RR_TP2"]

    # --- Setup 1: Retest (Только LONG) ---
    resistance_level = max(highs[-20:-3])
    if (ema50 > ema200 and closes[-1] > ema50 and closes[-3] > resistance_level and lows[-2] <= resistance_level and closes[-2] > resistance_level and closes[-1] > closes[-2]):
        sl_price = price - (atr_val * sl_multiplier)
        tp1_price = price + (price - sl_price) * tp1_rr
        tp2_price = price + (price - sl_price) * tp2_rr
        signal_data = ("LONG", price, sl_price, tp1_price, tp2_price, (resistance_level, price))

    support_level = min(lows[-20:-3])
    if not signal_data and ALLOW_SHORTS and (ema50 < ema200 and closes[-1] < ema50 and closes[-3] < support_level and highs[-2] >= support_level and closes[-2] < support_level and closes[-1] < closes[-2]):
        sl_price = price + (atr_val * sl_multiplier)
        tp1_price = price - (sl_price - price) * tp1_rr
        tp2_price = price - (sl_price - price) * tp2_rr
        signal_data = ("SHORT", price, sl_price, tp1_price, tp2_price, (price, support_level))

    # --- Setup 2: EMA/FVG Collision (Только LONG) ---
    if not signal_data:
        fvg_info = find_fvg(highs, lows)
        if fvg_info:
            fvg_type, fvg_low, fvg_high = fvg_info
            if fvg_low < ema200 < fvg_high and abs(price - ema200) / price < 0.003: # Increased tolerance
                entry_range = (fvg_low, fvg_high)
                if fvg_type == 'BULLISH' and price > ema200:
                    sl_price = price - (atr_val * sl_multiplier)
                    tp1_price = price + (price - sl_price) * tp1_rr
                    tp2_price = price + (price - sl_price) * tp2_rr
                    signal_data = ("LONG", price, sl_price, tp1_price, tp2_price, entry_range)
                elif fvg_type == 'BEARISH' and ALLOW_SHORTS and price < ema200:
                    sl_price = price + (atr_val * sl_multiplier)
                    tp1_price = price - (sl_price - price) * tp1_rr
                    tp2_price = price - (sl_price - price) * tp2_rr
                    signal_data = ("SHORT", price, sl_price, tp1_price, tp2_price, entry_range)

    # --- Final Check: OB Confirmation (Enhancer) ---
    if signal_data:
        ob_confirmed = False
        ob_info = find_confirmed_ob(closes, highs, lows, vols)
        if ob_info:
            ob_low, ob_high = ob_info
            entry_price = signal_data[1]
            if ob_low <= entry_price <= ob_high: ob_confirmed = True
        return signal_data + (ob_confirmed,)

    return None

# ================== BACKGROUND JOBS ==================
async def monitor_open_trades(context: ContextTypes.DEFAULT_TYPE):
    logging.info("Monitoring open trades...")
    try:
        with open(CSV_FILE, 'r', newline='') as f:
            trades = list(csv.reader(f))
        if not trades or len(trades) <= 1: return
    except (FileNotFoundError, StopIteration):
        return

    open_trades = [row for row in trades[1:] if row[7] == "OPEN"]
    if not open_trades:
        logging.info("No open trades to monitor.")
        return

    for trade in open_trades:
        date, symbol, side, entry, sl, tp1, tp2, result = trade
        price = get_current_price(symbol)
        if price is None: continue

        sl, tp1 = float(sl), float(tp1)
        trade_closed, new_result = False, ""

        logging.info(f"[Monitor] Checking {symbol} ({side}) | Price: {price:.4f}, SL: {sl:.4f}, TP: {tp1:.4f}")

        if side == "LONG" and (price <= sl or price >= tp1):
            trade_closed = True
            new_result = "WIN" if price >= tp1 else "LOSS"
        elif side == "SHORT" and (price >= sl or price <= tp1):
            trade_closed = True
            new_result = "WIN" if price <= tp1 else "LOSS"

        if trade_closed:
            update_trade(symbol, new_result)
            message = (f"<b>Trade Closed for {symbol}</b>\n\n"
                       f"<b>Result:</b> {new_result}\n"
                       f"<b>Side:</b> {side}\n"
                       f"<b>Entry Price:</b> {entry}\n"
                       f"<b>Closing Price:</b> {price:.4f}")
            logging.info(f"Closing trade for {symbol}. Result: {new_result}")
            for chat_id in context.job.data.get("chat_ids", []):
                await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')

async def run_analysis_and_monitoring(context: ContextTypes.DEFAULT_TYPE):
    """A single job that runs analysis and then monitors trades."""
    # Part 1: Monitoring (run this first to close trades and free up slots)
    await monitor_open_trades(context)

    # Part 2: Analysis
    if not CONFIG["ENABLED"]:
        return
        
    if not WATCHLIST:
        logging.warning("Watchlist is empty. Skipping analysis.")
        return

    open_trades_count = get_open_trades_count()
    if open_trades_count >= CONFIG["MAX_OPEN_TRADES"]:
        logging.info(f"Skipping analysis. Open trades ({open_trades_count}) reached limit ({CONFIG['MAX_OPEN_TRADES']}).")
        return

    logging.info(f"Running periodic analysis for symbols in watchlist: {WATCHLIST}")
    # Итерируемся по копии, чтобы избежать проблем с изменением списка во время итерации
    for symbol in WATCHLIST[:]:
        # Re-check limit before each analysis
        if get_open_trades_count() >= CONFIG["MAX_OPEN_TRADES"]:
            logging.info(f"Stopping analysis mid-run. Open trades limit reached.")
            break
            
        # --- Новая логика: Проверка "периода охлаждения" ---
        now = datetime.now()
        if symbol in SIGNAL_COOLDOWN:
            last_signal_time = SIGNAL_COOLDOWN[symbol]
            cooldown_delta = timedelta(minutes=CONFIG["COOLDOWN_PERIOD_MINUTES"])
            if now - last_signal_time < cooldown_delta:
                logging.info(f"Symbol {symbol} is in cooldown. Skipping.")
                continue
        
        try:
            signal_data = analyze(symbol)
            if signal_data:
                side, price, sl, tp1, tp2, entry_range, ob_confirmed = signal_data
                
                # --- Новая логика: Обновляем время охлаждения ---
                SIGNAL_COOLDOWN[symbol] = now
                
                signal_message = (f"<b>New Signal for {symbol}</b>\n\n"
                                  f"<b>Side:</b> {side}\n"
                                  f"<b>Entry Range:</b> {entry_range[0]:.4f} - {entry_range[1]:.4f}\n"
                                  f"<b>Entry Price:</b> {price:.4f}\n"
                                  f"<b>Stop Loss:</b> {sl:.4f}\n"
                                  f"<b>Take Profit 1 (RR {CONFIG['RR_TP1']}:1):</b> {tp1:.4f}\n"
                                  f"<b>Take Profit 2 (RR {CONFIG['RR_TP2']}:1):</b> {tp2:.4f}")
                if ob_confirmed: signal_message += "\n<b>Confirmation:</b> ✅ OB Confirmed"
                
                log_trade([datetime.now().strftime('%Y-%m-%d %H:%M'), symbol, side, price, sl, tp1, tp2, "OPEN"])
                
                for chat_id in context.job.data.get("chat_ids", []):
                    await context.bot.send_message(chat_id=chat_id, text=signal_message, parse_mode='HTML')
        except Exception as e:
            logging.error(f"Error during analysis of {symbol}: {e}")

async def update_watchlist_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Асинхронная задача для обновления глобального списка отслеживаемых монет.
    Запускает ресурсоемкий сканнер в отдельном потоке, чтобы не блокировать бота.
    """
    global WATCHLIST
    logging.info("Job: Starting market scan to update watchlist...")
    
    # Запускаем синхронную функцию сканера в отдельном потоке
    loop = asyncio.get_running_loop()
    new_watchlist = await loop.run_in_executor(
        None, market_scanner.scan_top_movers, 20
    )
    
    if new_watchlist:
        WATCHLIST = new_watchlist
        logging.info(f"Job: Watchlist updated successfully with {len(WATCHLIST)} symbols.")
        # Оповещаем администратора об обновлении
        message = "✅ **Watchlist Updated**\n\n" + "\n".join(WATCHLIST)
        for chat_id in context.job.data.get("chat_ids", []):
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
    else:
        logging.warning("Job: Market scan returned an empty list. Watchlist not updated.")


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
        await update.message.reply_text(f"✅ Risk set to: {CONFIG['RISK']}% ")
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
    status_text += f"\n<b>Watchlist ({len(WATCHLIST)} symbols):</b>\n" + ", ".join(WATCHLIST)
    await update.message.reply_html(status_text)

@restricted
async def get_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий список отслеживаемых монет."""
    if WATCHLIST:
        message = "📄 **Current Watchlist**\n\n" + "\n".join(WATCHLIST)
    else:
        message = "Watchlist is currently empty. The scanner might be running."
    await update.message.reply_text(message, parse_mode='Markdown')

@restricted
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open(CSV_FILE) as f:
            total = sum(1 for _ in f) - 1
        await update.message.reply_text(f"📊 Trades logged: {total}")
    except FileNotFoundError:
        await update.message.reply_text("Trade log file not found.")

@restricted
async def winrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with open(CSV_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            closed_trades = [row for row in reader if row[7] != "OPEN"]
    except (FileNotFoundError, StopIteration):
        await update.message.reply_text("Trade log file not found or is empty.")
        return

    if not closed_trades:
        await update.message.reply_text("No closed trades found to calculate winrate.")
        return

    wins = sum(1 for trade in closed_trades if trade[7] == "WIN")
    losses = sum(1 for trade in closed_trades if trade[7] == "LOSS")
    total_closed = wins + losses

    if total_closed == 0:
        await update.message.reply_text("No trades with WIN/LOSS status found.")
        return

    win_rate = (wins / total_closed) * 100

    message = (
        f"<b>Trade Performance</b>\n\n"
        f"Total Closed Trades: {total_closed}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"<b>Winrate: {win_rate:.2f}%</b>"
    )
    await update.message.reply_html(message)

@restricted
async def get_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends the trades.csv file to the user."""
    try:
        await update.message.reply_document(document=open(CSV_FILE, 'rb'), filename=CSV_FILE)
    except FileNotFoundError:
        await update.message.reply_text("trades.csv file not found.")

# ================== MAIN ==================
def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logging.error("Error: BOT_TOKEN environment variable not set.")
        return

    application = ApplicationBuilder().token(token).build()
    job_queue = application.job_queue

    # --- Новые задачи ---
    # 1. Задача обновления списка наблюдения (каждые 4 часа)
    job_queue.run_repeating(update_watchlist_job, interval=timedelta(hours=4), first=10, data={"chat_ids": ALLOWED_CHAT_IDS})
    
    # 2. Основная задача анализа и мониторинга (каждую минуту)
    job_queue.run_repeating(run_analysis_and_monitoring, interval=60, first=20, data={"chat_ids": ALLOWED_CHAT_IDS})

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('risk', risk))
    application.add_handler(CommandHandler('tf', timeframe))
    application.add_handler(CommandHandler('on', on))
    application.add_handler(CommandHandler('off', off))
    application.add_handler(CommandHandler('status', status))
    application.add_handler(CommandHandler('stats', stats))
    application.add_handler(CommandHandler('winrate', winrate))
    application.add_handler(CommandHandler('get_csv', get_csv))
    # Новая команда для просмотра списка
    application.add_handler(CommandHandler('watchlist', get_watchlist))


    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()
