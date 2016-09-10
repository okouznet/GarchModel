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
from fredapi import Fred

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
    test.testStationary(ts=stock_data['xlp'][1000:1200])

    df = np.column_stack((data['xlp'], data['GDP'], data['UMCSENT'], data['CPIAUCSL']))
    dftest = pd.DataFrame(df, columns=['xlp', 'GDP', 'UMCSENT', 'CPI'])
    test.CorrelationPlot(data=dftest)
    test.scatterMatrix(data=dftest)

    #MLE estimation
    m = model.Model()
    ylag = stock_data['xlp'].shift(periods=1)
    ylag = ylag.values
    for i in range(0, len(ylag)):
        ylag[i] = float(ylag[i])
    y = stock_data['xlp'].values
    for i in range(0, len(y)):
        y[i] = float(y[i])
    ylag[0] = 0
    matrix = np.column_stack((y,ylag))
    garch = model.GARCH(endog=matrix)
    param = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    results = garch.fit(start_params=param)
    print(results.params)

    #fitted values
    fitted = garch.fittedValues(endog=matrix, est_params=results.params)
    garch.plotFitted(y=y[1000:1200], fitted=fitted[1000:1200])

    #tests
    resid = y - fitted
    test.residualTest(residual=resid[1000:1200])
    test.acfTest(ts=resid[1000:1200])