from __future__ import division
import numpy as np
import statistics as st
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
        omega, alpha, beta, a, b = param
        n = len(endog[:, 0])
        r = endog[:, 0] - a - b * endog[:, 1]
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])  # GARCH(1,1) model

        return stats.lognorm.pdf(x=r, s=0.94, loc=0, scale=1)

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