from __future__ import division
import numpy as np
import statistics as st
import pandas as pd
import math
from matplotlib import  pyplot as plt
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

class Model:

    def __init__(self):
        self.est = []

    def ARCH11_logL(self, param, endog):
        a, b, sigma = param
        e = endog[:, 0] - a - b*endog[:, 1]
        e[1] = 0
        e = e[~np.isnan(e)]

        s = (sigma**2)/(1 - (b**2))
        h = (e/(sigma))
        logL = -((np.log(s) - h).sum())
       # print(logL)

        return (logL)

    def GARCH11_logLSimple(self, param, endog):
        omega, alpha, beta, a, b = param
        r = endog[:, 0] - a - b * endog[:, 1]
        n = len(r)
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])  # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

    def GARCH11_logL(self, param, endog):
        omega, alpha, beta, a, b, gdp = param
        r = endog[:, 0] - a - b*endog[:, 1] - gdp*endog[:, 2]
        n = len(r)
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])  # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

    def fittedValues(self, endog, est_params, simple_model):
        if(simple_model == True):
            omega, alpha, beta, a, b = est_params
            self.est = a + b * endog[:, 1]
            e = endog[:, 0] - self.est
        else:
            omega, alpha, beta, a, b, gdp = est_params
            self.est = a + b * endog[:, 1] + gdp*endog[:, 2]
            e = endog[:, 0] - self.est

        n = len(e)
        h = np.ones(n) * 0.01
        h[2] = st.variance(e[0:3])
        for t in range(1, n):
            h[t] = omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1])  # GARCH(1,1) model
        z = e / h
        return self.est

    def plotFitted(self, y, fitted):
        fitted_val = np.column_stack((y, fitted))
        graph = pd.DataFrame(fitted_val, columns=['actual', 'fitted'])
        graph.plot()
        plt.show()

