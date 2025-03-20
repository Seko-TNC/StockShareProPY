import yfinance as yf
import pandas as pd
import streamlit as st
import time

def get_all_tickers():
    # Get all S&P 500 stocks (can be expanded to other indices)
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return sp500["Symbol"].tolist()

def get_stock_data(tickers):
    # Use yf.download() for bulk requests to avoid rate limits
    stock_data = yf.download(tickers, period='1d', group_by='ticker', progress=False)
    close_prices = {}
    for ticker in tickers:
        try:
            if ticker in stock_data and not stock_data[ticker].empty:
                close_prices[ticker] = stock_data[ticker]['Close'].dropna().iloc[-1]
            else:
                close_prices[ticker] = None  # If no data, set None
        except KeyError:
            close_prices[ticker] = None
    return close_prices

def analyze_stock(ticker, current_price):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    
    # Extract relevant financial indicators
    pe_ratio = stock_info.get('trailingPE', None)
    eps = stock_info.get('trailingEps', None)
    fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', None)
    fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', None)
    dividend_yield = stock_info.get('dividendYield', None)
    market_cap = stock_info.get('marketCap', None)
    debt_to_equity = stock_info.get('debtToEquity', None)
    
    # Ensure necessary values are present
    if pe_ratio and eps and current_price:
        fair_value = eps * 15  # Assuming 15x P/E is fair
        target_price = eps * 18  # Assuming 18x P/E is an optimistic target
        discount = (fair_value - current_price) / fair_value
        
        # Confidence Bands
        if discount >= 0.25:  # More than 25% undervalued
            confidence = "Strong Buy"
        elif discount >= 0.15:  # 15-25% undervalued
            confidence = "Buy"
        elif discount >= 0.05:  # 5-15% undervalued
            confidence = "Watchlist"
        else:
            return None  # Skip if not undervalued enough
    else:
        return None
    
    return {
        "Ticker": ticker,
        "Current Price": current_price,
        "52-Week High": fifty_two_week_high,
        "52-Week Low": fifty_two_week_low,
        "P/E Ratio": pe_ratio,
        "EPS": eps,
        "Fair Value": fair_value,
        "Target Price": target_price,
        "Discount %": round(discount * 100, 2),
        "Confidence Band": confidence,
        "Dividend Yield": dividend_yield,
        "Market Cap": market_cap,
        "Debt to Equity": debt_to_equity
    }

def main():
    st.title("Stock Buy Recommendations")
    st.write("This tool identifies buy opportunities based on financial indicators and categorizes them into confidence bands.")
    
    tickers = get_all_tickers()  # Fetch all S&P 500 tickers
    
    # Option to limit number of stocks analyzed
    stock_limit = st.selectbox("Select number of stocks to analyze:", [50, 100, 200, len(tickers)], index=0)
    tickers = tickers[:stock_limit]  # Limit stocks for efficiency
    
    if st.button("Run Analysis"):
        results = []
        skipped = []  # Track skipped stocks
        
        st.write("Fetching stock prices...")
        stock_prices = get_stock_data(tickers)  # Get all stock prices at once
        
        st.write("Analyzing stocks...")
        for ticker in tickers:
            current_price = stock_prices.get(ticker)
            if current_price is None or pd.isna(current_price):
                skipped.append(ticker)
                continue
            
            result = analyze_stock(ticker, current_price)
            if result:
                results.append(result)
            else:
                skipped.append(ticker)  # Log skipped tickers
            
            time.sleep(0.5)  # Delay to avoid hitting API limits
        
        total_stocks = len(tickers)
        reviewed_stocks = total_stocks - len(skipped)
        
        st.write(f"Total stocks analyzed: {total_stocks}")
        st.write(f"Stocks reviewed: {reviewed_stocks}")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Export option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "stock_analysis.csv", "text/csv")
        else:
            st.write("No buy opportunities found.")
        
        # Log skipped stocks for debugging
        if skipped:
            st.write(f"Skipped {len(skipped)} stocks due to missing data. First 10 skipped: {', '.join(skipped[:10])}")

if __name__ == "__main__":
    main()
