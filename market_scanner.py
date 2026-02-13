import requests
import logging
from operator import itemgetter

# Настройка логирования для сканера
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
scanner_logger = logging.getLogger(__name__)


def get_klines(symbol, timeframe='4h', limit=10):
    """
    Запрашивает K-линии (свечи) для указанного символа.
    """
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
    try:
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        # Увеличим таймаут для надежности
        r = requests.get(url, params=params, timeout=15).json()
        if r.get("code") == 0:
            return r.get("data", [])
        else:
            scanner_logger.warning(f"API error for {symbol}: {r.get('msg')}")
            return []
    except requests.RequestException as e:
        scanner_logger.error(f"Scanner: Request error fetching klines for {symbol}: {e}")
        return []

def scan_top_movers(num_symbols=20):
    """
    Сканирует все символы из файла, рассчитывает их оценку на основе волатильности
    и объема, а затем возвращает список самых перспективных.
    """
    symbols_file = "bingx_futures_symbols.txt"
    try:
        with open(symbols_file, 'r') as f:
            # Убираем пустые строки, если они есть
            all_symbols = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        scanner_logger.error(f"Scanner: Symbol file not found at '{symbols_file}'.")
        return []

    scanner_logger.info(f"Scanner: Starting scan of {len(all_symbols)} symbols...")
    scored_symbols = []

    for i, symbol in enumerate(all_symbols):
        # Логируем прогресс каждые 50 символов
        if i > 0 and i % 50 == 0:
            scanner_logger.info(f"Scanner: Scanned {i}/{len(all_symbols)} symbols...")

        # Получаем последние 2 свечи (4-часовые) для сравнения
        klines = get_klines(symbol, timeframe='4h', limit=2)
        if len(klines) < 2:
            continue

        try:
            last_candle = klines[-1]
            prev_candle = klines[-2]

            open_price = float(last_candle['open'])
            close_price = float(last_candle['close'])
            volume = float(last_candle['volume'])
            prev_volume = float(prev_candle['volume'])

            # 1. Волатильность (абсолютное процентное изменение)
            volatility = abs(close_price - open_price) / open_price

            # 2. Всплеск объема (насколько текущий объем больше предыдущего)
            # Добавляем малое число, чтобы избежать деления на ноль
            volume_spike_ratio = volume / (prev_volume + 1e-9)

            # 3. Простая оценка: волатильность, умноженная на коэффициент всплеска объема.
            # Это дает приоритет монетам, которые не просто движутся, но и делают это с поддержкой объема.
            score = volatility * volume_spike_ratio
            
            if score > 0:
                scored_symbols.append({'symbol': symbol, 'score': score})

        except (ValueError, KeyError, IndexError) as e:
            scanner_logger.warning(f"Scanner: Could not process data for {symbol}: {e}")
            continue
    
    if not scored_symbols:
        scanner_logger.warning("Scanner: No symbols could be scored.")
        return []

    # Сортируем символы по убыванию оценки и берем топ N
    sorted_symbols = sorted(scored_symbols, key=itemgetter('score'), reverse=True)
    top_symbols = [s['symbol'] for s in sorted_symbols[:num_symbols]]
    
    scanner_logger.info(f"Scanner: Scan complete. Top {len(top_symbols)} movers: {top_symbols}")
    
    return top_symbols

if __name__ == '__main__':
    # Этот блок позволяет запустить сканер напрямую для теста
    print("Running market scanner directly for testing...")
    top_movers = scan_top_movers()
    print("\nTop movers found:")
    print(top_movers)
