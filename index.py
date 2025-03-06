import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import schedule
import time
import smtplib
import yfinance as yf
import streamlit as st
from email.mime.text import MIMEText
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# 📌 Cargar y limpiar datos
def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato de archivo no compatible.")
        df = df.drop_duplicates().dropna()
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None

# 📌 Análisis de mercadeo y ventas
def conversion_rate(sales, visitors):
    return (sales / visitors) * 100

def roi(revenue, cost):
    return (revenue - cost) / cost * 100

def predict_sales(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + 31).reshape(-1, 1)
    future_sales = model.predict(future_days)
    return future_days.flatten(), future_sales

# 📌 Análisis de mercados bursátiles
def get_stock_data(ticker, start="2023-01-01", end="2024-01-01"):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    return df

def predict_stock_prices(df, days=30):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# 📌 Automatización de reportes
def generate_report():
    df = load_data("data/sales_data.csv")
    print("📊 Reporte de Ventas Generado")
    print(df.describe())

def scheduled_report():
    schedule.every().day.at("08:00").do(generate_report)
    while True:
        schedule.run_pending()
        time.sleep(60)

# 📌 Alertas por email
def send_alert(email, message):
    msg = MIMEText(message)
    msg['Subject'] = "📉 Alerta de Mercado"
    msg['From'] = "tuemail@gmail.com"
    msg['To'] = email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login("tuemail@gmail.com", "tupassword")
        server.sendmail("tuemail@gmail.com", email, msg.as_string())

def check_stock_alerts(df, email):
    rsi = (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(window=14).mean() / 
           df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(window=14).mean()).fillna(50)
    if rsi.iloc[-1] < 30:
        send_alert(email, "⚠️ Acción en sobreventa. Posible oportunidad de compra.")
    elif rsi.iloc[-1] > 70:
        send_alert(email, "⚠️ Acción en sobrecompra. Posible caída de precio.")

# 📌 Interfaz gráfica con Streamlit
def run_dashboard():
    st.title("📊 Dashboard de Análisis")
    st.sidebar.header("Seleccionar análisis")
    option = st.sidebar.selectbox("Elija una opción", ["Ventas", "Mercados Bursátiles"])
    
    if option == "Ventas":
        df = load_data("data/sales_data.csv")
        st.write("Datos de ventas:", df.head())
        future_days, future_sales = predict_sales(df)
        st.line_chart(future_sales)
    elif option == "Mercados Bursátiles":
        ticker = st.text_input("Ingrese el ticker de la acción:", "AAPL")
        df = get_stock_data(ticker)
        st.line_chart(df["Close"])

# 📌 Menú principal (CLI)
def main():
    print("📊 Bienvenido al sistema de análisis 📈")
    
    while True:
        print("\nSeleccione una opción:")
        print("1. Análisis de ventas")
        print("2. Análisis de mercados bursátiles")
        print("3. Generar informes")
        print("4. Ejecutar automatización")
        print("5. Abrir Dashboard")
        print("6. Salir")

        choice = input("Ingrese su opción: ")

        if choice == '1':
            sales = float(input("Ingrese las ventas: "))
            visitors = float(input("Ingrese los visitantes: "))
            revenue = float(input("Ingrese el ingreso total: "))
            cost = float(input("Ingrese el costo total: "))
            print(f"Tasa de Conversión: {conversion_rate(sales, visitors):.2f}%")
            print(f"ROI: {roi(revenue, cost):.2f}%")

        elif choice == '2':
            ticker = input("Ingrese el ticker de la acción: ")
            df = get_stock_data(ticker)
            forecast = predict_stock_prices(df)
            print(f"Predicción de precios: {forecast}")

        elif choice == '3':
            generate_report()

        elif choice == '4':
            print("Ejecutando automatización...")
            scheduled_report()

        elif choice == '5':
            run_dashboard()

        elif choice == '6':
            print("Saliendo del programa...")
            break

        else:
            print("Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()
