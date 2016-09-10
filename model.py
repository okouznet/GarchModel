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
        pass

    def GARCH11_logL(self, param, y, ylag):
        omega, alpha, beta, a, b = param
        n = len(y)
        r = y - a - b*ylag
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])  # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

    def garch_pmf(self, endog, param):
        omega, alpha, beta, a, b, sigma = param
        n = len(endog[:, 0])
        e = endog[:, 0] - a - b * endog[:, 1]
        h = np.ones(n) * 0.01
        h[0] = sigma
        for t in range(1, n):
            h[t] = omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1])  # GARCH(1,1) model

        return stats.lognorm.pdf(x=e, s=0.94, loc=0, scale=h) #FIX

class GARCH(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)

        super(GARCH, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        m = Model()

        return -np.log(m.garch_pmf(endog=self.endog, param=params))

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        return super(GARCH, self).fit(start_params=start_params,
                                                    maxiter=maxiter, maxfun=maxfun, **kwds)

    def fittedValues(self, endog, est_params):
        omega, alpha, beta, a, b, sigma = est_params
        n = len(endog[:, 0])
        est = a + b*endog[:, 1]
        e = est - a - b * endog[:, 1]
        h = np.ones(n) * 0.01
        h[0] = sigma
        for t in range(1, n):
            h[t] = omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1])  # GARCH(1,1) model
        z = e * h
        return est + z

    def plotFitted(self, y, fitted):
        fitted_val = np.column_stack((y, fitted))
        graph = pd.DataFrame(fitted_val, columns=['actual', 'fitted'])
        graph.plot()
        plt.show()