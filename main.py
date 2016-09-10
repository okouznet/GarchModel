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
    #stock_data.to_csv()
    stock_data = pd.read_csv('stocks.csv')

    #get shock data
    #shock_data = d.shockData()
    #shock_data.to_csv(path_or_buf='shocks.csv', sep=',')
    shock_data = pd.read_csv('shocks.csv')

    data = pd.merge(stock_data, shock_data, how='outer', on='Dates')
   # pd.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #data.to_csv(path_or_buf='data.csv', sep=',')
    #test = tests.Test()
    #test.CorrelationPlot(data=data)

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

    fitted = garch.fittedValues(endog=matrix, est_params=results.params)

    fitted_val = np.column_stack((y, fitted))
    graph = pd.DataFrame(fitted_val)
    #graph.plot()
    #plt.show()
    resid = y - fitted
    ts1 = pd.Series(np.random.randn(1000))
    #pd.tools.plotting.autocorrelation_plot(y)

    # o = optimize.fmin(m.GARCH11_logL, np.array([.1, .1, .1, .1, .1]), args=(y, ylag), full_output=1)
    # R = np.abs(o[0])
    # print("omega = %.6f\nbeta  = %.6f\nalpha = %.6f\na = %.6f\nb = %.6f\n" % (R[0], R[2], R[1], R[3]. R[4]))
    # print(R)