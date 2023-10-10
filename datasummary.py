import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn.metrics as metrics
#dspring = pd.read_csv("springforwardtotal.csv",index_col='Date',parse_dates=True)
#dfall = pd.read_csv("Falltotal.csv",index_col='Date',parse_dates=True)
#dwinter = pd.read_csv("wintertotal.csv",index_col='Date',parse_dates=True)

#plt.plot(Date, Forward total)
#plt.xlabel('Time (hr)')
#plt.ylabel('Forward Total (kwh)')
#plt.show()


rspring=pd.read_csv("springresult.csv")
rfall=pd.read_csv("Fallresult.csv")
rwinter=pd.read_csv("winterresult.csv")
#springsummary=dspring.describe()
#fallsummary=dfall.describe()
#wintersummary=dwinter.describe()
#springresult=rspring.describe()
#fallresult=rfall.describe()
#winterresult=rwinter.describe()
#print(rspring.describe())

#print(rfall.describe())
#print(rwinter.describe())

mspring=pd.read_csv("springame.csv")
plt.plot(mspring['Unit'],mspring['MAE'],'o')
plt.title("Spring Data Mean Absolute Error VS Unit ID")
plt.xlabel('Unit')
plt.ylabel("MAE")
plt.show()
plt.savefig("Spring Data Mean Absolute Error VS Unit ID")

mfall=pd.read_csv("fallmae.csv")
plt.plot(mfall['Unit'],mfall['MAE'],'o')
plt.title("Fall Data Mean Absolute Error VS Unit ID")
plt.xlabel('Unit')
plt.ylabel("MAE")
plt.show()
plt.savefig("Fall Data Mean Absolute Error VS Unit ID")

mwinter=pd.read_csv("wintermae.csv")

plt.plot(mwinter['Unit'],mwinter['MAE'],'o')
plt.title("Winter Data Mean Absolute Error VS Unit ID")
plt.xlabel('Unit')
plt.ylabel("MAE")
plt.show()
plt.savefig("Winter Data Mean Absolute Error VS Unit ID")




"""
rspring2=pd.read_csv("springresult2.csv")
rfall2=pd.read_csv("fallresult2.csv")
rwinter2=pd.read_csv("winterresult2.csv")
rspring4=pd.read_csv("springresult4.csv")
rfall4=pd.read_csv("fallresult4.csv")
rwinter4=pd.read_csv("winterresult4.csv")


plt.figure()
rspring2.boxplot()
plt.title("Spring Result Summary Boxplot")
plt.savefig("Spring Result Summary Boxplot")

plt.figure()
rfall2.boxplot()
plt.title("Fall Result Summary Boxplot")
plt.savefig("Fall Result Summary Boxplot")

plt.figure()
rwinter2.boxplot()
plt.title("Winter Result Summary Boxplot")
plt.savefig("Winter Result Summary Boxplot")


from pandas.plotting import scatter_matrix
plt.figure()
scatter_matrix(rspring4, alpha=0.2, figsize=(6, 6))
plt.title("Spring Feature-Feature Relationships Result Summary")
plt.savefig("spring scatter")

plt.figure()
scatter_matrix(rfall4, alpha=0.2, figsize=(6, 6))
plt.title("Fall Feature-Feature Relationships Result Summary")
plt.savefig("fall scatter")

plt.figure()
scatter_matrix(rwinter4, alpha=0.2, figsize=(6, 6))
plt.title("Winter Feature-Feature Relationships Result Summary")
plt.savefig("winter scatter")




plt.figure()
springresult.groupby('Unit').plas.hist(alpha=0.4)
plt.title("Spring Result Hist")
plt.savefig("spring Hist")


print(dspring.describe())
print(dfall.describe())
print(dwinter.describe())
print(rspring.describe())
print(rfall.describe())
print(rwinter.describe()
"""