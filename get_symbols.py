import requests

def get_bingx_usdt_futures():
    """
    Fetches USDT-margined perpetual futures symbols from BingX and saves them to a file.
    """
    url = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
    output_filename = "bingx_futures_symbols.txt"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == 0:
            contracts = data.get("data", [])
            usdt_symbols = [
                contract['symbol'] 
                for contract in contracts 
                if contract['symbol'].endswith('-USDT')
            ]
            
            with open(output_filename, 'w') as f:
                for symbol in usdt_symbols:
                    f.write(f"{symbol}\n")
            
            print(f"Successfully wrote {len(usdt_symbols)} symbols to {output_filename}")

        else:
            print(f"API returned an error: {data.get('msg')}")

    except requests.RequestException as e:
        print(f"An error occurred during the request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_bingx_usdt_futures()