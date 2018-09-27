import pandas as pd
import numpy as np
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from seasonal import fit_seasons, adjust_seasons
import statsmodels.api as sm
from fbprophet import Prophet
series = pd.read_csv('../../Downloads/gams.txt', header=0,usecols=[0,1])
series.columns=['ds','y']
print (series.head())
#statsmodel

#result = seasonal_decompose(series.values,freq=1, model='additive')
#series=np.array(series.values)
#seasons, trend = fit_seasons(series)
#adjusted = adjust_seasons(s, seasons=seasons)
#residual = adjusted - trend
#dta = series.resample("M").fillna(method="ffill")
#res=sm.tsa.seasonal_decompose(dta)
#fig=res.plot()
#pyplot.tight_layout()

#print(result.trend)
#print(result.seasonal)
#print(result.resid)
#print(result.observed)

#series.rolling(10).mean().plot(figsize=(20,10), linewidth=1, fontsize=20)
#series.plot()
#pyplot.xlabel('Coize', fontsize=5);
#pyplot.show()


#propheti
m = Prophet()
m.fit(series);
forecast = m.predict(series)
m.plot_components(forecast);
