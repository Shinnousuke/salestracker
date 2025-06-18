import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

st.title("📈 Sales Forecasting App")

# File Upload
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Sales' columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        st.success("File loaded successfully.")
        
        st.write("📊 Sample Data")
        st.write(df.head())

        # Plot original data
        st.subheader("Sales Over Time")
        fig, ax = plt.subplots()
        df['Sales'].plot(ax=ax)
        ax.set_title("Historical Sales Data")
        st.pyplot(fig)

        # Forecasting period
        steps = st.slider("Select number of months to forecast", 1, 24, 6)

        st.subheader("SARIMAX Model Training...")
        model = SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        st.success("Model training complete.")

        # Forecasting
        forecast = results.get_forecast(steps=steps)
        forecast_index = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        forecast_df = pd.DataFrame({
            'Forecast': forecast.predicted_mean,
            'Lower CI': forecast.conf_int().iloc[:, 0],
            'Upper CI': forecast.conf_int().iloc[:, 1]
        }, index=forecast_index)

        # Plot forecast
        st.subheader("Forecasted Sales")
        fig2, ax2 = plt.subplots()
        df['Sales'].plot(ax=ax2, label='Historical')
        forecast_df['Forecast'].plot(ax=ax2, label='Forecast')
        ax2.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='gray', alpha=0.3)
        ax2.legend()
        ax2.set_title("Sales Forecast with SARIMAX")
        st.pyplot(fig2)

        st.subheader("Forecasted Values")
        st.write(forecast_df)
        
    except Exception as e:
        st.error(f"Error: {e}")
