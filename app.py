import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ================= PAGE SETTINGS =================
st.set_page_config(page_title="Gen-AI Stock Analyst", layout="wide")
st.title("📈 Gen-AI Fundamental + Technical Stock Analyst (Free & No API Key)")

# ================= USER INPUT =================
stock = st.text_input("Enter Stock Ticker (Example: RELIANCE.NS / AAPL)")

# ================= LOAD LIGHT AI MODEL =================
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return nlp

# ================= BUTTON ACTION =================
if st.button("Analyze Stock"):

    if stock == "":
        st.warning("Please enter a stock ticker")
    else:
        ticker = yf.Ticker(stock)

        # ================= FUNDAMENTALS =================
        info = ticker.info

        name = info.get("longName", "N/A")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", "N/A")
        rev = info.get("totalRevenue", "N/A")
        profit = info.get("grossProfits", "N/A")
        pe = info.get("trailingPE", "N/A")
        pb = info.get("priceToBook", "N/A")
        roe = info.get("returnOnEquity", "N/A")
        debt = info.get("debtToEquity", "N/A")

        st.subheader("🏢 Company & Financial Summary")
        st.write(f"""
        **Name:** {name}  
        **Sector:** {sector}  
        **Industry:** {industry}  
        **Market Cap:** {market_cap}  
        **Revenue:** {rev}  
        **Profit:** {profit}  
        **PE Ratio:** {pe}  
        **PB Ratio:** {pb}  
        **ROE:** {roe}  
        **Debt to Equity:** {debt}  
        """)

        # ================= NEWS =================
        st.subheader("📰 Latest News Headlines")

        news_text = ""
        try:
            news = ticker.news
            if news and len(news) > 0:
                for n in news[:5]:
                    title = n.get("title", "No Title Available")
                    source = n.get("publisher", "Unknown Source")
                    st.write(f"- **{title}** ({source})")
                    news_text += title + ". "
            else:
                st.info("No recent news available.")
                news_text = "No major recent news."
        except:
            st.info("News data could not be fetched.")
            news_text = "No major recent news."

        # ================= TECHNICALS =================
        st.subheader("📊 Technical Indicators")

        try:
            data = ticker.history(period="1y")

            if data is None or data.empty:
                st.info("No technical data available.")
                sma_signal = "N/A"
                rsi_signal = "N/A"
                macd_signal = "N/A"
                rsi = 0
            else:
                close = data["Close"]

                # SMA
                data["SMA_50"] = close.rolling(50).mean()
                data["SMA_200"] = close.rolling(200).mean()

                if data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]:
                    sma_signal = "Bullish Trend"
                else:
                    sma_signal = "Bearish Trend"

                # RSI
                rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
                if rsi > 70:
                    rsi_signal = "Overbought"
                elif rsi < 30:
                    rsi_signal = "Oversold"
                else:
                    rsi_signal = "Neutral"

                # MACD
                macd = MACD(close)
                macd_value = macd.macd().iloc[-1]
                signal_value = macd.macd_signal().iloc[-1]

                if macd_value > signal_value:
                    macd_signal = "Bullish Momentum"
                else:
                    macd_signal = "Bearish Momentum"

                st.write(f"**SMA Trend:** {sma_signal}")
                st.write(f"**RSI:** {rsi:.2f} → {rsi_signal}")
                st.write(f"**MACD:** {macd_signal}")

        except:
            st.info("Technical indicator calculation failed.")
            sma_signal = "N/A"
            rsi_signal = "N/A"
            macd_signal = "N/A"
            rsi = 0

        # ================= AI ANALYSIS =================
        st.subheader("🤖 Gen-AI Stock Analysis")

        with st.spinner("Generating AI-based analysis..."):
            nlp = load_model()

            prompt = f"""
            Provide a short stock analysis summary based on:
            Company: {name}
            Sector: {sector}
            PE: {pe}
            ROE: {roe}
            Debt: {debt}

            News: {news_text}

            Technicals:
            SMA: {sma_signal}
            RSI: {rsi_signal}
            MACD: {macd_signal}

            Summarize into:
            Company Overview, Financial Health, Risk, and Verdict.
            """

            result = nlp(prompt, max_length=250)[0]["generated_text"]

        st.success(result)
