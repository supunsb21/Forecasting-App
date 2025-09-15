import streamlit as st
import pandas as pd
from prophet import Prophet
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Function to load the pre-trained model and make predictions
def load_and_forecast(df, months_to_forecast, model_path):
    model = joblib.load(model_path)
    future = model.make_future_dataframe(periods=months_to_forecast, freq='M')
    forecast = model.predict(future)
    return forecast

# Function to load the dataset based on the department
def load_data(department):
    if department == "TATA Commercial Workshop":
        return pd.read_csv('Srv_&_Bodyshop-TATA_Commercial_monthly_complaints.csv', parse_dates=['Month_Year'])
    elif department == "MB Service & Bodyshop":
        return pd.read_csv('Service_&_Bodyshop-MB_monthly_complaints.csv', parse_dates=['Month_Year'])
    elif department == "M&M Service":
        return pd.read_csv('M&M_Service_monthly_complaints.csv', parse_dates=['Month_Year'])
    else:
        return pd.DataFrame()

# Streamlit UI
st.set_page_config(page_title="Complaints Forecasting", layout="wide")
st.title("Complaints Forecasting")
st.markdown("""
This application utilizes the **Prophet model** to forecast the future number of complaints in different departments.
Select a department and visualize both historical data and future predictions.
""")

# Sidebar
with st.sidebar:
    st.header("Select Department")
    department = st.selectbox("Choose Department", ["TATA Commercial Workshop", "MB Service & Bodyshop", "M&M Service"], index=0)
    months_to_forecast = st.slider("Select the number of months to forecast", 1, 12, 6)

# Load data and model
df = load_data(department)
if df.empty:
    st.warning("No data loaded. Please check the selected department or file paths.")
    st.stop()

model_path = {
    "TATA Commercial Workshop": 'TATA_model.pkl',
    "MB Service & Bodyshop": 'MB_model.pkl',
    "M&M Service": 'MnM_model.pkl'
}.get(department, '')

forecast = load_and_forecast(df, months_to_forecast, model_path)

# Prepare data
df = df.rename(columns={'Month_Year': 'ds', 'Complaint Count': 'y'})
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m')
df['ds_str'] = df['ds'].dt.strftime('%Y-%m')

forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast['ds_str'] = forecast['ds'].dt.strftime('%Y-%m')

# Evaluation
historical_forecast = forecast[forecast['ds'] <= df['ds'].iloc[-1]]
y_true = df.set_index('ds')['y']
y_pred = historical_forecast.set_index('ds')['yhat']
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Forecast table
forecast_df = forecast[['ds_str', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months_to_forecast)
forecast_df = forecast_df.rename(columns={
    'ds_str': 'Month',
    'yhat': 'Forecasted Value',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
})

# Line chart
actual_series = df.set_index('ds_str')['y']
forecast_series = forecast[forecast['ds'] > df['ds'].iloc[-1]].set_index('ds_str')['yhat']
combined_df = pd.DataFrame({
    'Actual Data': actual_series,
    'Forecast Data': forecast_series
})

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Model Evaluation Metrics")
    st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.markdown(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

st.subheader("Complaints Forecasting")
st.line_chart(combined_df)

st.subheader("Forecasted Values Table")
st.dataframe(forecast_df, use_container_width=True)
