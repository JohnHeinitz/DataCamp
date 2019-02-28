import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Read in the data
data = pd.read_csv("Budget_test.csv", index_col=0)

# print(data.head())
data.index = pd.to_datetime(data.index)
data.columns = ['WRVU Production']

plt.plot(data)
plt.ylabel('wrvus')
plt.show()

autocorrelation_plot(data)
pyplot.show()

model = ARIMA(data, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

x = data.values
size = int(len(x)*0.5)
train, test = x[0:size], x[size:len(x)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.33f' % error)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

