from statsmodels.base.model import GenericLikelihoodModel

import model
import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

if __name__ == '__main__':
    d = data.Data()

    #get stock data
    #stock_data = d.getStocks(stocks=['xlp','xly', 'xle', 'xlk', 'xlf', 'xlv', 'xli', 'xlb', 'xlre', 'xlu'])
    #stock_data.Dates = pd.to_datetime(stock_data['Dates'], format='%Y-%m-%d')
    #stock_data.set_index(['Dates'], inplace=True)
    #stock_data.to_csv(path_or_buf='stocks.csv', sep=',')
    #calculate percent change
    #stock_data = stock_data.pct_change(periods=1)

    stock_data = pd.read_csv('stocks.csv')

    #stock_data.plot()
    #plt.show()

    #get shock data
    """
    gdp = d.getShocks(shock='GDP')
    umcsent = d.getShocks(shock='UMCSENT')
    cpi = d.getShocks(shock='CPIAUCSL')

    shock_data = pd.merge(gdp, umcsent, how='outer', on='dates')
    shock_data = shock_data.sort_values('dates')
    shock_data = pd.merge(shock_data, cpi, how='outer', on='dates')
    shock_data = shock_data.sort_values('dates')

    shock_data.dates = pd.to_datetime(shock_data['dates'], format='%Y-%m-%d')
    shock_data.set_index(['dates'], inplace=True)

    shock_data = shock_data.fillna(0)
    shock_data['GDP'] = shock_data['GDP'].astype(float)
    shock_data['UMCSENT'] = shock_data['UMCSENT'].astype(float)
    shock_data['CPIAUCSL'] = shock_data['CPIAUCSL'].astype(float)
    #shock_data = shock_data.pct_change(periods=1)

    shock_data.to_csv(path_or_buf='shocks.csv', sep=',')

    shock_data.plot()
    plt.show()
    """
    shock_data = pd.read_csv('shocks.csv')
    #shock_data.plot()
    #plt.show()
    m = model.Model()
    ylag = stock_data['xlp'].shift(periods=1)
    ylag = ylag.values
    for i in range(0, len(ylag)):
        ylag[i] = float(ylag[i])
    y = stock_data['xlp'].values
    for i in range(0, len(y)):
        y[i] = float(y[i])
    ylag[0] = 0
    o = optimize.fmin(m.GARCH11_logL, np.array([.1, .1, .1, .1, .1]), args=(y, ylag), full_output=1)
    R = np.abs(o[0])
    #print("omega = %.6f\nbeta  = %.6f\nalpha = %.6f\na = %.6f\nb = %.6f\n" % (R[0], R[2], R[1], R[3]. R[4]))
    print(R)

    matrix = np.column_stack((y,ylag))
    garch = model.GARCH(endog=matrix)
    param = [0.1, 0.1, 0.1, 0.1, 0.1]
    results = garch.fit(start_params=param)
    print(results.params)

    #m.garch_pmf(param=[.1, .1, .1, .1, .1], y=y, ylag=ylag)


