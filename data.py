import pandas as pd
import numpy as np
import statistics as st
from yahoo_finance import Share
from fredapi import Fred

class Data:
    def __init__(self):
        pass

    def getStocks(self, stocks=[]):
        data = pd.DataFrame()
        dates=[]
        for stock in stocks:
            s = Share(stock)
            d = pd.DataFrame(s.get_historical('2007-01-01', '2016-06-01'))
            dates = d['Date']
            data[stock] = d['Adj_Close']
            data[stock] = data[stock].astype(float)
        data['Dates'] = dates
        return data

    def getShocks(self, shock):
        fred = Fred(api_key='2d4b5ef2420e64d7e42928ffd9d6ff8c')
        d = pd.DataFrame(fred.get_series_as_of_date(shock, '1/1/2007'))
        data = pd.DataFrame()
        data['dates'] = d['realtime_start']
        data[shock] = d['value']
        data = data.sort_values('dates')
        data = data.drop_duplicates(subset='dates')
        return data


    def GARCH11_logL(self, param, r):
        omega, alpha, beta = param
        n = len(r)
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])  # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

