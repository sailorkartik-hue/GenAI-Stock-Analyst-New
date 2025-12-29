import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.set_page_config(page_title="Gen-AI Stock Analyst", layout="wide")
st.title("📈 Gen-AI Fundamental + Technical Stock Analyst (Streamlit Safe Version)")

stock = st.text_input("Enter Stock Ticker (Example: RELIANCE.NS / AAPL)")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return nlp

if st.button("Analyze Stock"):
    if stock == "":
        st.warning("Please enter a stock ticker")
    else:
        ticker = yf.Ticker(stock)

        # =================== FUNDAMENTALS (SAFE MODE) ===================
        st.subheader("🏢 Company & Financial Summary")

        try:
            fast = ticker.fast_info
            name = stock
            sector = "Not Available (Yahoo Restricted)"
            industry = "Not Available (Yahoo Restricted)"

            market_cap = fast.get("market_cap", "N/A")
            last_price = fast.get("last_price", "N/A")
            fifty_two_high = fast.get("year_high", "N/A")
            fifty_two_low = fast.get("year_low", "N/A")

            st.write(f"""
            **Stock:** {name}  
            **Sector:** {sector}  
            **Industry:** {industry}  
            **Last Price:** {last_price}  
            **Market Cap:** {market_cap}  
            **52W High:** {fifty_two_high}  
            **52W Low:** {fifty_two_low}  
            """)
        except:
            st.error("Yahoo Finance blocked fundamental data temporarily 😞")
            market_cap = "N/A"
            last_price = "N/A"

        # =================== NEWS ===================
        st.subheader("📰 Latest News Headlines")
        news_text = ""

        try:
            news = ticker.news
            if news and len(news) > 0:
                for n in news[:5]:
                    title = n.get("title", "No Title")
                    source = n.get("publisher", "Unknown")
                    st.write(f"- **{title}** ({source})")
                    news_text += title + ". "
            else:
                st.info("News not available for this stock.")
                news_text = "No recent news available."
        except:
            st.info("Yahoo blocked news temporarily.")
            news_text = "No news available."

        # =================== TECHNICALS ===================
        st.subheader("📊 Technical Indicators")

        try:
            data = ticker.history(period="1y")

            if data is None or data.empty:
                st.info("No price history available.")
                sma_signal = "N/A"
                rsi_signal = "N/A"
                macd_signal = "N/A"
                rsi = 0
            else:
                close = data["Close"]

                data["SMA_50"] = close.rolling(50).mean()
                data["SMA_200"] = close.rolling(200).mean()

                if data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]:
                    sma_signal = "Bullish Trend"
                else:
                    sma_signal = "Bearish Trend"

                rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
                if rsi > 70:
                    rsi_signal = "Overbought"
                elif rsi < 30:
                    rsi_signal = "Oversold"
                else:
                    rsi_signal = "Neutral"

                macd = MACD(close)
                macd_value = macd.macd().iloc[-1]
                signal_value = macd.macd_signal().iloc[-1]

                macd_signal = "Bullish Momentum" if macd_value > signal_value else "Bearish Momentum"

                st.write(f"**SMA Trend:** {sma_signal}")
                st.write(f"**RSI:** {rsi:.2f} → {rsi_signal}")
                st.write(f"**MACD:** {macd_signal}")

        except:
            st.info("Technical indicator calculation failed.")
            sma_signal = "N/A"
            rsi_signal = "N/A"
            macd_signal = "N/A"
            rsi = 0

        # =================== AI ANALYSIS ===================
        st.subheader("🤖 Gen-AI Stock Analysis")

        with st.spinner("Generating AI-based analysis..."):
            nlp = load_model()

            prompt = f"""
            Create a stock summary.
            Price info: Market Cap={market_cap}, Last Price={last_price}
            News: {news_text}
            Technicals: SMA={sma_signal}, RSI={rsi_signal}, MACD={macd_signal}
            Provide: Overview, Risk, Sentiment, Verdict.
            """

            result = nlp(prompt, max_length=250)[0]["generated_text"]

        st.success(result)
