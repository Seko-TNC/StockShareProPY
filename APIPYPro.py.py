import yfinance as yf
import pandas as pd
import streamlit as st
import time
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_all_tickers():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return sp500["Symbol"].tolist()

def get_stock_data(tickers):
    stock_data = yf.download(tickers, period='1y', group_by='ticker', progress=False)
    close_prices = {}
    for ticker in tickers:
        try:
            if ticker in stock_data and not stock_data[ticker].empty:
                close_prices[ticker] = stock_data[ticker]['Close'].dropna().iloc[-1]
            else:
                close_prices[ticker] = None
        except KeyError:
            close_prices[ticker] = None
    return close_prices

def analyze_stock(ticker, current_price):
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    hist = stock.history(period='1y')
    
    if hist.empty or len(hist) < 50:
        return None
    
    # Technical Indicators
    hist['50_MA'] = hist['Close'].rolling(50).mean()
    hist['200_MA'] = hist['Close'].rolling(200).mean()
    hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
    upper, middle, lower = talib.BBANDS(hist['Close'], timeperiod=20)
    hist['Upper_Band'], hist['Middle_Band'], hist['Lower_Band'] = upper, middle, lower
    
    # Extract fundamental indicators
    pe_ratio = stock_info.get('trailingPE', None)
    eps = stock_info.get('trailingEps', None)
    pb_ratio = stock_info.get('priceToBook', None)
    roe = stock_info.get('returnOnEquity', None)
    fcf = stock_info.get('freeCashflow', None)
    revenue_growth = stock_info.get('revenueGrowth', None)
    debt_to_equity = stock_info.get('debtToEquity', None)
    dividend_yield = stock_info.get('dividendYield', None)
    
    # Calculate Fair Value
    if pe_ratio and eps:
        fair_value = eps * 15
        target_price = eps * 18
        discount = (fair_value - current_price) / fair_value
    else:
        return None
    
    # Confidence Bands
    if discount >= 0.25:
        confidence = "Strong Buy"
    elif discount >= 0.15:
        confidence = "Buy"
    elif discount >= 0.05:
        confidence = "Watchlist"
    else:
        return None
    
    # Technical Buy Signals
    technical_signal = "Neutral"
    if hist['RSI'].iloc[-1] < 30 and current_price < hist['Lower_Band'].iloc[-1]:
        technical_signal = "Buy"
    elif hist['50_MA'].iloc[-1] > hist['200_MA'].iloc[-1]:
        technical_signal = "Bullish"
    elif hist['50_MA'].iloc[-1] < hist['200_MA'].iloc[-1]:
        technical_signal = "Bearish"
    
    return {
        "Ticker": ticker,
        "Current Price": current_price,
        "Fair Value": fair_value,
        "Target Price": target_price,
        "Discount %": round(discount * 100, 2),
        "Confidence Band": confidence,
        "P/E Ratio": pe_ratio,
        "P/B Ratio": pb_ratio,
        "ROE": roe,
        "Free Cash Flow": fcf,
        "Revenue Growth": revenue_growth,
        "Debt to Equity": debt_to_equity,
        "Dividend Yield": dividend_yield,
        "Technical Signal": technical_signal
    }

def train_ml_model():
    # Placeholder dataset
    data = pd.DataFrame({
        "P/E": np.random.uniform(5, 30, 1000),
        "P/B": np.random.uniform(0.5, 5, 1000),
        "ROE": np.random.uniform(5, 25, 1000),
        "Debt/Equity": np.random.uniform(0, 3, 1000),
        "Revenue Growth": np.random.uniform(-0.1, 0.3, 1000),
        "Target": np.random.choice([0, 1], 1000)  # 1 = Good Buy, 0 = Avoid
    })
    
    X = data.drop(columns=["Target"])
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

def main():
    st.title("Stock Buy Recommendations")
    st.write("This tool identifies investment opportunities using fundamental & technical analysis.")
    
    tickers = get_all_tickers()
    stock_limit = st.selectbox("Select number of stocks to analyze:", [50, 100, 200, len(tickers)], index=0)
    tickers = tickers[:stock_limit]
    
    model, scaler = train_ml_model()
    
    if st.button("Run Analysis"):
        results = []
        skipped = []
        stock_prices = get_stock_data(tickers)
        
        for ticker in tickers:
            current_price = stock_prices.get(ticker)
            if current_price is None or pd.isna(current_price):
                skipped.append(ticker)
                continue
            
            stock_data = analyze_stock(ticker, current_price)
            if stock_data:
                X_pred = scaler.transform([[
                    stock_data["P/E Ratio"],
                    stock_data["P/B Ratio"],
                    stock_data["ROE"],
                    stock_data["Debt to Equity"],
                    stock_data["Revenue Growth"]
                ]])
                prediction = model.predict(X_pred)[0]
                stock_data["ML Recommendation"] = "Good Buy" if prediction == 1 else "Avoid"
                results.append(stock_data)
            else:
                skipped.append(ticker)
            
            time.sleep(0.5)
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "stock_analysis.csv", "text/csv")
        else:
            st.write("No buy opportunities found.")
        
        if skipped:
            st.write(f"Skipped {len(skipped)} stocks due to missing data. First 10 skipped: {', '.join(skipped[:10])}")

if __name__ == "__main__":
    main()
