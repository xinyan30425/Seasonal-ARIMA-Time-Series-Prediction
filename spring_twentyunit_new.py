import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn.metrics as metrics
dspring = pd.read_csv("wintertotal40.csv",index_col='Date',parse_dates=True)

def dfgroupby(field,csvpath):#参数说明：field分组的字段
    #dfall= pd.read_csv("Spring_forwardtotal.csv")
    df_groups = dspring.groupby(field)
    result=[]#结果列表
    unit_id_list = []
    #遍历分组列表，存df到结果列表
    for name,df in df_groups:
        result.append(df[['Forward total']])
        unit_id_list.append(name)
    return result,unit_id_list
if __name__ == '__main__':
    #调用
    df_list,unit_id_list=dfgroupby("ID","wintertotal40.csv")
    for df in df_list:
        print(df)

#seasonal decompose
#for df in df_list:
for i in range(len(df_list)):
#for i in range(5):
    df = df_list[i]
    unit_id = unit_id_list[i]
    result = seasonal_decompose(df,model='multiplicative', period=7)

    result.plot()
    #plt.figure()
    # plt.title("Seasonal decompose Unit%i"%unit_id)
    plt.title("Seasonal Decompose Unit " + str(unit_id))
    plt.show()
    #plt.savefig("Seasonal decompose Unit " + str(unit_id))


# Import the library
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

####################### model fitting ################
# Fit auto_arima function to dataset
dict_order = {}
dict_seasonal_order = {}
dict_MAE = {}
dict_MSE = {}
dict_RMSE= {}
dict_Rsquare={}

for i in range(len(df_list)):
#for i in range(5):
    df = df_list[i]
    unit_id = unit_id_list[i]
    stepwise_fit = auto_arima(df, start_p=1, start_q=1,
                          max_p=10, max_q=10, m=7,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',  # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)  # set to stepwise
    stepwise_fit.summary()
    order= stepwise_fit.order
    dict_order[unit_id] = order
    seasonal_order= stepwise_fit.seasonal_order
    dict_seasonal_order[unit_id] = seasonal_order
    train = df.iloc[:len(df) - 14]
    test = df.iloc[len(df) - 14:]  # set two weeks (14 days) for testing
    plt.figure()
    axes = plt.gca()
    test.plot(legend=True)
    plt.title("Seasonal Decompose Unit " + str(unit_id))
    plt.xlabel('Date')
    plt.ylabel('Electricity Usage(kwh)')
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)
    axes.title.set_size(12)
    plt.savefig("Winter Seasonal Decompose Unit " + str(unit_id).format(i))
    plt.show()
    print(dict_order)
    print(dict_seasonal_order)

#dict1 = {"number of storage arrays": 45, "number of ports": 2390}
# Fit a SARIMAX(0, 1, 1)x(2, 1, 1, 7) on the training set
# model = SARIMAX(train,order=(1, 1, 1),seasonal_order=(2, 1, 1, 7))

    model = SARIMAX(train,order=(order),seasonal_order=(seasonal_order))
    trainresult = model.fit()
    trainresult.summary()

# Predictions against the test set
    start = len(train)
    end = len(train) + len(test) - 1
    predictions = trainresult.predict(start, end,
                                 typ='levels').rename("Predictions")

# plot predictions and actual values
    plt.figure()
    axes = plt.gca()
    plt.plot(predictions)
    plt.plot(test)
    plt.xlabel('Date')
    plt.ylabel('Electricity Usage(kwh)')
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)
    plt.legend(["Predictions", "Test"])
    # predictions.plot(legend=True)
    # test.plot(legend=True)
    plt.title("WinterTest VS Prediction Unit "+ str(unit_id))
    axes.title.set_size(12)
    plt.savefig("Winter Test VS Prediction Unit " + str(unit_id).format(i))
    plt.show()

# Train the model on the full dataset
    df2 = df.iloc[6:len(df)]
    modelfull = model = SARIMAX(df,
                            order=(order),
                            seasonal_order=(seasonal_order))
    resultforcast = modelfull.fit()

    # Forecast for the next one month
    forcast = resultforcast.predict(start=7,
                                     end=(len(df)),
                                     typ='levels').rename('Forcast')
    # end=(len(Spring3) - 1) + 30,
    # Plot the forcast values
    plt.figure()
    axes = plt.gca()
    plt.plot(df2)
    plt.plot(forcast)
    plt.legend(["Actual Data", "Forcast"])
    # predictions.plot(legend=True)
    # test.plot(legend=True)
    plt.xlabel('Date')
    plt.ylabel('Electricity Usage(kWh)')
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)
    plt.title("Winter Actual Data VS Forcast Unit " + str(unit_id))
    axes.title.set_size(12)
    plt.savefig("Winter Actual Data VS Forcast Unit " + str(unit_id).format(i))
    plt.show()
    # Winter3.plot(figsize=(12, 5), legend=True)
    # forecast.plot(legend=True)
    # test.plot(legend=True)

#Forecast for next 30 days

    modelfull = model = SARIMAX(df,
                                order=(order),
                                seasonal_order=(seasonal_order))
    resultforecast30 = modelfull.fit()

    # Forecast for the next one month
    forecast30 = resultforecast30.predict(start=7,
                                     end=(len(df)+30),
                                     typ='levels').rename('Forecast30')
    # end=(len(Spring3) - 1) + 30,
    # Plot the forecast values
    plt.figure()
    axes = plt.gca()
    plt.plot(df2)
    plt.plot(forecast30)
    plt.legend(["Actual Data", "Forecast"])
    # predictions.plot(legend=True)
    # test.plot(legend=True)
    plt.xlabel('Date')
    plt.ylabel('Electricity Usage(kWh)')
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)
    plt.title("Winter Actual Data VS Forcast and 30 Days Forecast Unit " + str(unit_id))
    axes.title.set_size(12)
    plt.savefig("Winter Actual Data VS Forcast Unit and 30 Days Forecast Unit" + str(unit_id).format(i))
    plt.show()




    y = df2
    yhat = forcast
    x = list(range(len(df2)))
    plt.figure()
    axes = plt.gca()
    plt.scatter(x, y, color="blue", label="Original")
    plt.plot(x, yhat, color="red", label="Forcast")
    plt.xlabel('Date(day)')
    plt.ylabel('Electricity Usage(kWh)')
    axes.xaxis.label.set_size(12)
    axes.yaxis.label.set_size(12)
    plt.title("Winter Accuracy Analysis Actual VS Prediction Unit "+ str(unit_id))
    axes.title.set_size(12)
    plt.legend(["Actual Data", "Prediction"])
    plt.savefig("Winter Accuracy Analysis Actual VS Prediction Unit "+ str(unit_id).format(i))
    plt.show()
    mae = metrics.mean_absolute_error(y, yhat)
    mse = metrics.mean_squared_error(y, yhat)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = metrics.r2_score(y, yhat)
    #maeString = str(mae)
    #mseString = str(mse)
    #rmseString = str(rmse)
    #r2String = str(r2)
    dict_MAE[unit_id] = mae
    dict_MSE[unit_id] = mse
    dict_RMSE[unit_id] = rmse
    dict_Rsquare[unit_id] = r2
    print("Results of sklearn.metrics:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)

####################### result analysis ################
# dco = pd.DataFrame(data=dict_order, index=[2])
dco = pd.DataFrame(data=dict_order)
dco = (dco.T)
print(dco)
dco.to_excel('dict_order.xlsx')

# dcso = pd.DataFrame(data=dict_seasonal_order, index=[0,1,2])
dcso = pd.DataFrame(data=dict_seasonal_order)
dcso = (dcso.T)
print(dcso)
dcso.to_excel('dict_seasonal_order.xlsx')

dmae = pd.DataFrame(data=dict_MAE, index=[0])
dmae = (dmae.T)
print(dmae)
dmae.to_excel('MAE.xlsx')

dmse = pd.DataFrame(data=dict_MSE, index=[0])
dmse = (dmse.T)
print(dmse)
dmse.to_excel('MSE.xlsx')

#drmse = pd.DataFrame(data=dict_RMSE, index=[0])
#drmse = (drmse.T)
#print(drmse)
#rmse.to_excel('RMSE.xlsx')

dr2 = pd.DataFrame(data=dict_Rsquare, index=[0])
dr2 = (dr2.T)
print(dr2)
dr2.to_excel('r2.xlsx')

