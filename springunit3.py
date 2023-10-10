import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

Spring3 = pd.read_csv('springunit1forwardtotal.csv',
                      index_col='Date',
                      parse_dates=True)
Spring3.head()
result = seasonal_decompose(Spring3,
                            model='multiplicative')
# ETS plot
result.plot()
plt.title('Seasonal decompose')
plt.show()

# Import the library
from pmdarima import auto_arima

# Ignore harmless warnings
import warnings

warnings.filterwarnings("ignore")

# Fit auto_arima function to dataset
stepwise_fit = auto_arima(Spring3, start_p=1, start_q=1,
                          max_p=3, max_q=3, m=7,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',  # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)  # set to stepwise

# To print the summary
stepwise_fit.summary()
order= stepwise_fit.order
seasonal_order= stepwise_fit.seasonal_order

# Split data into train / test sets
train = Spring3.iloc[:len(Spring3) - 14]
test = Spring3.iloc[len(Spring3) - 14:]  # set one week (7 days) for testing

# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 7) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train,
                order=(order),
                seasonal_order=(seasonal_order))
result = model.fit()
result.summary()

start = len(train)
end = len(train) + len(test)-1

# Predictions for one-week against the test set
predictions = result.predict(start, end,
                             typ='levels').rename("Predictions")

# plot predictions and actual values
plt.plot(predictions)
plt.plot(test)
plt.legend(["predictions", "test"])
#predictions.plot(legend=True)
#test.plot(legend=True)
plt.title("test vs prediction")
plt.show()

# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
rmse(test, predictions)

# Calculate mean squared error
mean_squared_error(test, predictions)

# Train the model on the full dataset
model = model = SARIMAX(Spring3,
                        order=(order),
                        seasonal_order=(seasonal_order))
resultforcast = model.fit()

# Forecast for the next one month
forecast = resultforcast.predict(start=1,
                          end=(len(Spring3)),
                          typ='levels').rename('Forecast')
#end=(len(Spring3) - 1) + 30,
# Plot the forecast values
plt.plot(Spring3)
plt.plot(forecast)
plt.legend(["Spring3", "forecast"])
#predictions.plot(legend=True)
#test.plot(legend=True)
plt.title("Actual vs Forcast")
plt.show()
#Spring3.plot(figsize=(12, 5), legend=True)
#forecast.plot(legend=True)
import sklearn.metrics as metrics
y=Spring3
yhat=forecast
x = list(range(len(Spring3)))
plt.scatter(x, y, color="blue", label="original")
plt.plot(x,yhat, color="red", label="predicted")
plt.legend()
plt.show()
mae = metrics.mean_absolute_error(y, yhat)
mse = metrics.mean_squared_error(y, yhat)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(y,yhat)

print("Results of sklearn.metrics:")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-Squared:", r2)
