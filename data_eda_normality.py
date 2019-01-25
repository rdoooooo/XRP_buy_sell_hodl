from seaborn import distplot
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import pickle
import statsmodels as sm
import statsmodels.graphics.tsaplots as smgraphicstsa
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from collections import defaultdict
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
# Load data
data_path = 'xrp_data_clean.csv'
df = pd.read_csv(data_path)
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

df.head()


df = df.Close.pct_change()
df.dropna(inplace=True)
df.head()

'''
To apply AR models, need to check how stationary the signal is
'''


def plot_dt(df):
    (df * 100).plot()
    plt.ylabel('Daily % Change')
    plt.ylabel('Date')


plot_dt(df)
'''
Plotting Rolling Statistics: We can plot the moving average or moving
variance and see if it varies with time. By moving average/variance I
mean that at any instant ‘t’, we’ll take the average/variance of the last
year, i.e. last 12 months. But again this is more of a visual technique.

Dickey-Fuller Test: This is one of the statistical tests for checking
stationarity. Here the null hypothesis is that the TS is non-stationary.
The test results comprise of a Test Statistic and some Critical Values for
difference confidence levels. If the ‘Test Statistic’ is less than the
‘Critical Value’, we can reject the null hypothesis and say that
the series is stationary.

'''


def test_stationarity(df, window=10):

    # Determing rolling statistics
    timeseries = df.values
    rolmean = df.rolling(window).mean()
    rolstd = df.rolling(window).std()
    # Plot rolling statistics:
    orig = df.plot(color='blue', label='Original')
    mean = rolmean.plot(ax=orig, color='red', label='Rolling Mean')
    std = rolstd.plot(ax=orig, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(df, window=100)


# Look at pacf to see which lag terms should be signicant
'''
From Kaggle NB
q: determined by when the acf first crosses zero
p: determined by when the pacf have significant values
'''


def plot_cf(df, lags=np.arange(1, 20)):
    plt.figure()
    plt.subplot(211)
    smgraphicstsa.plot_acf(x=df, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    smgraphicstsa.plot_pacf(x=df, ax=plt.gca(), lags=lags)


plot_cf(df)


# Use Q-Q plot to test for normality

qqplot(df, fit=True, line='45', dist=stats.t)


# histogram
distplot(df, bins=100)
