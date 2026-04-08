import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")

# ------------------ LOAD DATA ------------------
try:
    df = pd.read_csv("EW-MAX.csv")
except:
    st.error("❌ File not found! Check your file path.")
    st.stop()

# ------------------ DATA PREVIEW ------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df)

# ------------------ CHECK REQUIRED COLUMNS ------------------
required_cols = ["Open", "High", "Low", "Volume", "Close"]

if not all(col in df.columns for col in required_cols):
    st.error("❌ CSV must contain columns: Open, High, Low, Volume, Close")
    st.stop()

# ------------------ ADD DAY COLUMN IF MISSING ------------------
if "Day" not in df.columns:
    df["Day"] = range(1, len(df) + 1)

# ------------------ DATA PREPROCESSING ------------------
st.subheader("🧹 Data Preprocessing")

col1, col2 = st.columns(2)

with col1:
    st.write("Missing Values:")
    st.write(df.isnull().sum())

with col2:
    st.write("Preview:")
    st.dataframe(df.head())

# ------------------ FEATURE ENGINEERING ------------------
st.subheader("⚙ Feature Engineering")
df["PriceRange"] = df["High"] - df["Low"]
st.dataframe(df.head())

# ------------------ METRICS ------------------
st.subheader("📌 Quick Stats")
col1, col2, col3 = st.columns(3)

col1.metric("Mean Close", f"{df['Close'].mean():.2f}")
col2.metric("Max Close", f"{df['Close'].max():.2f}")
col3.metric("Min Close", f"{df['Close'].min():.2f}")

# ------------------ VISUALIZATION ------------------
st.subheader("📉 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    plt.plot(df["Day"], df["Close"])
    plt.xlabel("Day")
    plt.ylabel("Close Price")
    plt.title("Closing Price Trend")
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    plt.scatter(df["Volume"], df["Close"])
    plt.xlabel("Volume")
    plt.ylabel("Close Price")
    plt.title("Volume vs Close")
    st.pyplot(fig2)

# ------------------ CORRELATION ------------------
st.subheader("🔗 Correlation Matrix")
st.dataframe(df.corr(numeric_only=True))

# ------------------ MODEL TRAINING ------------------
st.subheader("🤖 Model Training")

X = df[["Day", "Open", "High", "Low", "Volume", "PriceRange"]]
y = df["Close"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()

with st.spinner("Training model..."):
    model.fit(X_train, y_train)

st.success("Model trained successfully!")

# ------------------ MODEL EVALUATION ------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("📊 Model Evaluation")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("📥 User Input")

day = st.sidebar.slider("Day", 1, 365, 11)
open_price = st.sidebar.number_input("Open Price", value=110.0)
high = st.sidebar.number_input("High Price", value=115.0)
low = st.sidebar.number_input("Low Price", value=105.0)
volume = st.sidebar.number_input("Volume", value=2500)

# ------------------ PREDICTION ------------------
st.subheader("🎯 Predict Stock Price")

if st.button("Predict"):
    price_range = high - low
    input_data = np.array([[day, open_price, high, low, volume, price_range]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Closing Price: {prediction[0]:.2f}")