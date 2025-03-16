import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

file_path = "C:\\Users\\chandrashekar\\Downloads\\stores_sales_forecasting.csv"  # Change to your actual file path
df = pd.read_csv(file_path, encoding="latin1")

df["Order Date"] = pd.to_datetime(df["Order Date"])

df_sales = df.groupby("Order Date")["Sales"].sum().reset_index()

df_sales.set_index("Order Date", inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df_sales.index, df_sales["Sales"], label="Daily Sales", color="blue", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.title("Sales Trend Over Time")
plt.legend()
plt.grid(True)
plt.show()
decomposition = seasonal_decompose(df_sales["Sales"], model="additive", period=30)
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.show()

model = ARIMA(df_sales["Sales"], order=(5,1,0)) 
model_fit = model.fit()

forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

future_dates = pd.date_range(start=df_sales.index[-1], periods=forecast_steps+1, freq="D")[1:]

plt.figure(figsize=(12, 6))
plt.plot(df_sales.index, df_sales["Sales"], label="Actual Sales", color="blue", alpha=0.7)
plt.plot(future_dates, forecast, label="Forecasted Sales", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.title("Sales Forecast for the Next 30 Days")
plt.legend()
plt.grid(True)
plt.show()

forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Sales": forecast.values})
forecast_df.to_csv("sales_forecast.csv", index=False)

print("Forecasting complete! The results are saved in 'sales_forecast.csv'.")
