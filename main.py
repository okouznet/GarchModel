from statsmodels.base.model import GenericLikelihoodModel

import model
import data
import tests
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy import optimize
import statsmodels.api as sm


from biokit.viz import corrplot

if __name__ == '__main__':
    d = data.Data()
    #stock_data = d.stockData(stocks=['xlp','xly', 'xle', 'xlk', 'xlf', 'xlv', 'xli', 'xlb', 'xlre', 'xlu'])
    #shock_data = d.shockData()
    stock_data = pd.read_csv('stocks.csv')
    shock_data = pd.read_csv('shocks.csv')

    data = pd.merge(stock_data, shock_data, how='outer', on='Dates')
    test = tests.Test()

    #basic statistics test
    #test.testStationary(ts=stock_data['xlp'][1000:1200])

    df = np.column_stack((data['xlp'], data['GDP'], data['UMCSENT'], data['CPIAUCSL']))
    dftest = pd.DataFrame(df, columns=['xlp', 'GDP', 'UMCSENT', 'CPI'])
    #test.CorrelationPlot(data=dftest)
    #test.scatterMatrix(data=dftest)

    m = model.Model()
    ylag = data['xlp'].shift(periods=1)
    ylag = ylag.values
    for i in range(0, len(ylag)):
        ylag[i] = float(ylag[i])
    y = data['xlp'].values
    for i in range(0, len(y)):
        y[i] = float(y[i])

    gdp = data['GDP']
    gdp = gdp.fillna(0)
    #ylag[0:2] = 0

    y = y[2:2370]
    ylag = ylag[2:2370]
    gdp = gdp[2:2370]
    matrix = np.column_stack((y, ylag, gdp))


    simple = optimize.fmin(m.GARCH11_logLSimple,np.array([.5, .5, .5, .5, .5]), args=(matrix,), maxfun=100000, maxiter=10000, full_output=1)
    full = optimize.fmin(m.GARCH11_logL,np.array([.5, .5, .5, .5, .5, .5]), args=(matrix,), maxfun=100000, maxiter=10000, full_output=1)
    simple_est = np.abs(simple[0])
    full_est = np.abs(full[0])
    #print(sim)

    simple_fitted = m.fittedValues(endog=matrix, est_params=simple_est, simple_model=True)
    full_fitted = m.fittedValues(endog=matrix, est_params=full_est, simple_model=False)

    m.plotFitted(y=y[1000:1200], fitted=simple_fitted[1000:1200])
    m.plotFitted(y=y[1000:1200], fitted=full_fitted[1000:1200])

    # tests
    #resid = y - fitted
    #test.residualTest(residual=resid[1000:1200])
    #test.acfTest(ts=resid[1000:1200])
    test.ttest(a=simple_fitted, b=full_fitted)
