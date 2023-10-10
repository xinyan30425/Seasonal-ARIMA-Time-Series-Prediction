from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.stattools import acf,pacf

# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA



#df = pd.read_csv('September.csv',delim_whitespace=True, parse_dates=[1], infer_datetime_format=True)
#plt.plot(df['Date'],df["Forward total"])



col_list = ["Date", "Forwardtotal"]
df = pd.read_csv('unit3 forward total.csv', usecols=col_list)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True) #set date as index
df.head()
df=df.dropna()


plt.xlabel("Date",fontsize=5)
plt.ylabel("Forwardtotal",fontsize=10)
plt.title("Unit 3 Forward Total consumption")
plt.plot(df)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model='multiplicative',period = int(len(df)/2))
result.plot()
plt.show()

#decomposition = sm.tsa.seasonal_decompose(df, model='additive',freq=38, extrapolate_trend = 38)
#fig = decomposition.plot()
#plt.show()


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    # perform dickey fuller test
    print("Results of dickey fuller test")
    adft = adfuller(timeseries['Forwardtotal'], autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)


test_stationarity(df)

df_log = np.log(df)
moving_avg = df_log.rolling(10).mean()
std_dev = df_log.rolling(10).std()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.plot(std_dev, color="black")
plt.show()
#eliminate trends out of a series and obtain a more stationary series
df_log_moving_avg_diff = df_log-moving_avg
df_log_moving_avg_diff.dropna(inplace=True)



test_stationarity(df_log_moving_avg_diff)
weighted_average = df_log.ewm(halflife=10, min_periods=0,adjust=True).mean()


df_log_diff = df_log - df_log.shift()
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Forwardtotal")
plt.plot(df_log_diff) #Let us test the stationarity of our resultant series
df_log_diff.dropna(inplace=True)
test_stationarity(df_log_diff)

from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_log, model='additive', freq = '10')
result.plot()
plt.show()
trend = result.trend
trend.dropna(inplace=True)
seasonality = result.seasonal
seasonality.dropna(inplace=True)
residual = result.resid
residual.dropna(inplace=True)
x = residual.to_frame()
residualnew=x.rename(columns={"resid": "Forwardtotal"})
test_stationarity(residualnew)

#Finding the best parameters for our model
#Autocorrelation Function(ACF)


# we use d value here(data_log_shift)
acf = acf(df_log_diff, nlags=15)
pacf= pacf(df_log_diff, nlags=15,method='ols')#plot PACF
plt.subplot(121)
plt.plot(acf)
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()#plot ACF
plt.subplot(122)
plt.plot(pacf)
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()


#fitting model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(3,1,3))
result_AR = model.fit(disp = 0)
plt.plot(df_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title("sum of squares of residuals")
print('RSS : %f' %sum((result_AR.fittedvalues-df_log_diff["Forwardtotal"])**2))

#prediction
result_AR.plot_predict(1,500)
x=result_AR.forecast(steps=200)


#forecast electricity consumption for next 4 months

future=df_log
future=future.reset_index()
mon=future["Date"]
mon=mon+pd.DateOffset(months=7)
future_dates = mon[-7-1:]
future = future.set_index('Date')
newDf = pd.DataFrame(index=future_dates, columns=future.columns)
future = pd.concat([future,newDf])
future["Forecast Consumption"]= result_AR.predict(start=35, end =43, dynamic=True)
future["Forecast Consumption"].iloc[-10:]=result_AR.forecast(steps=10)[0]
future[['Forwardtotal','Forecast Consumption']].plot()


# we founded the predicted values in the above code and we need to print the values in the form of series
ARIMA_predicts = pd.Series(result_AR.fittedvalues,copy=True)
ARIMA_predicts.head()

# finding the cummulative sum
ARIMA_predicts_cumsum = ARIMA_predicts.cumsum()
print(ARIMA_predicts_cumsum.head())


ARIMA_predicts_log = pd.Series(df_log['Forwardtotal`'],index = df_log.index)
ARIMA_predicts_log = ARIMA_predicts_log.add(ARIMA_predicts_cumsum,fill_value=0)
print(ARIMA_predicts_log.head())

# converting back to the exponential form results in getting back to the original data.
ARIMA_final_preditcs = np.exp(ARIMA_predicts_log)
rcParams['figure.figsize']=10,10
plt.plot(df)
plt.plot(ARIMA_predicts_cumsum)

plt.plot(ARIMA_predicts_cumsum)
plt.plot(df)

#future prediction
result_AR.plot_predict(1,500)
x=result_AR.forecast(steps=200)


