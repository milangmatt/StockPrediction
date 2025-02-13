import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_nifty100_tickers():
    """Retrieves the tickers from the NIFTY 100 index from NSE India website."""
    try:
        url = "https://www.nseindia.com/indices/broad-market-indices/nifty-100"  # NSE Nifty 100 page
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, "html.parser")

        # Find the table containing NIFTY 100 constituents (inspect the NSE website's HTML)
        # The exact table structure might change, so inspect and adjust the selectors if needed.
        table = soup.find("table", {"id": "dataTable"}) # Example: look for a table with id 'dataTable'
        if table is None:
            table = soup.find("table", {"class": "table table-bordered"}) # Example: look for a table with class 'table table-bordered'

        if table is None:
            raise ValueError("Could not find the NIFTY 100 table on the NSE website. Check the website structure for the correct table selector.")

        tickers = []
        for row in table.find_all("tr")[1:]:  # Skip the header row
            columns = row.find_all("td")
            if columns:
                symbol = columns[1].text.strip() # Example: symbol is in the second column (index 1)
                tickers.append(symbol + ".NS") # Add ".NS" suffix for NSE tickers

        return tickers

    except requests.exceptions.RequestException as e:
        print(f"Error fetching NSE website: {e}")
        return None
    except ValueError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_top_n_by_market_cap(tickers, num_tickers=100):
    """Gets the top N tickers by market cap from a list."""
    tickers_with_market_cap = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).info
            market_cap = data.get('marketCap')
            if market_cap is not None:
                tickers_with_market_cap.append((ticker, market_cap))
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    tickers_with_market_cap.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in tickers_with_market_cap[:num_tickers]]


nifty100_tickers = get_nifty100_tickers()

if nifty100_tickers:
    top_100_by_market_cap = get_top_n_by_market_cap(nifty100_tickers)
    print(top_100_by_market_cap)
else:
    print("Could not retrieve NIFTY 100 tickers.")


# Note:
# 1. Install necessary libraries: pip install yfinance beautifulsoup4 requests
# 2. The code now fetches tickers directly from the NSE website.
# 3. Web scraping is fragile.  The NSE website structure might change, requiring
#    updates to the table selectors.  Inspect the page's HTML to find the correct
#    elements.
# 4. Error handling is improved.
# 5. Consider a professional data feed for the most reliable approach.
# 6. Market cap data might have gaps.  Error handling is included.