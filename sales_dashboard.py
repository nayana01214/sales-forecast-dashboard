import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Custom CSS for dark mode + white text
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, .stSidebar, .css-18e3th9 {
        color: white !important;
    }
    .stDownloadButton button, .stButton button {
        background-color: #21c8f6;
        color: black;
        border-radius: 8px;
    }
    h1 {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #ff4b4b, #f9cb28, #21c8f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #ff4b4b; }
        to { text-shadow: 0 0 20px #21c8f6; }
    }
    </style>
""", unsafe_allow_html=True)

# Animated Title
st.markdown("<h1>📈 Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
forecast_period = st.sidebar.slider("Months to Forecast", 1, 24, 12)

# File upload
uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("👋 Please upload a CSV file with columns: 'data', 'venda', 'estoque', 'preco'.")
    st.stop()

# Load and preprocess
df = pd.read_csv(uploaded_file)
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

# Monthly sales grouping
monthly_sales = df.resample('M', on='data').sum().reset_index()
monthly_sales = monthly_sales[['data', 'venda']]
monthly_sales.columns = ['ds', 'y']

# Prophet model
model = Prophet()
model.fit(monthly_sales)

future = model.make_future_dataframe(periods=forecast_period, freq='M')
forecast = model.predict(future)

# Forecast Plot
st.subheader("🔮 Sales Forecast")
fig1 = model.plot(forecast)
fig1.set_facecolor('#0e1117')
fig1.axes[0].set_facecolor('#0e1117')
fig1.axes[0].tick_params(colors='white')
fig1.axes[0].yaxis.label.set_color('white')
fig1.axes[0].xaxis.label.set_color('white')
st.pyplot(fig1)

# Components Plot
st.subheader("📊 Trend and Seasonality")
fig2 = model.plot_components(forecast)
for ax in fig2.axes:
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
st.pyplot(fig2)

# Forecasted Values Table + Download
st.subheader("📥 Forecasted Values")
forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
st.dataframe(forecast_data.style.set_properties(**{'background-color': '#0e1117', 'color': 'white'}))

csv = forecast_data.to_csv(index=False)
st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

# Price and Stock Trends Plot
st.subheader("📈 Price and Stock Over Time")
fig3, ax = plt.subplots()
ax.plot(df['data'], df['preco'], label='Price', color='lime')
ax.set_ylabel("Price", color='lime')
ax2 = ax.twinx()
ax2.plot(df['data'], df['estoque'], label='Stock', color='cyan')
ax2.set_ylabel("Stock", color='cyan')
fig3.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax2.tick_params(colors='white')
fig3.legend(['Price', 'Stock'])
st.pyplot(fig3)

# Top 3 Sales Months Table
st.subheader("🏆 Top 3 Sales Months")
top_months = monthly_sales.sort_values('y', ascending=False).head(3)
top_months['ds'] = top_months['ds'].dt.strftime('%B %Y')
st.dataframe(top_months[['ds', 'y']].rename(columns={'ds': 'Month', 'y': 'Sales'}).style.set_properties(**{'background-color': '#0e1117', 'color': 'white'}))

# Footer
st.markdown("---")
st.markdown("<center>Built with by Nayana | Powered by Prophet + Streamlit</center>", unsafe_allow_html=True)
