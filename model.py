from __future__ import division
import numpy as np
import statistics as st
import pandas as pd
import math
from matplotlib import  pyplot as plt
import matplotlib

class Model:

    def __init__(self):
        self.est = []
        self.z = []

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

    def GARCH11_logLSeasonal(self, param, endog):
        omega, alpha, beta, a, b, monday, jan, end, gmon, gjan, gend = param
        r = endog[:, 0] - a - b*endog[:, 1] - monday*endog[:, 2] - jan*endog[:, 3] - end*endog[:, 4]
        n = len(r)
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = (omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])) * math.exp(gmon*endog[i, 2] - gjan*endog[i, 3] - gend*endog[i, 4]) # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

    def GARCH11_logLFull(self, param, endog):
        omega, alpha, beta, a, b, monday, jan, end, gmon, gjan, gend, gdp = param
        r = endog[:, 0] - a - b*endog[:, 1] - monday*endog[:, 2] - jan*endog[:, 3] - end*endog[:, 4] - gdp*endog[:, 5]
        n = len(r)
        s = np.ones(n) * 0.01
        s[2] = st.variance(r[0:3])
        for i in range(3, n):
            s[i] = (omega + alpha * r[i - 1] ** 2 + beta * (s[i - 1])) * math.exp(gmon*endog[i, 2] - gjan*endog[i, 3] - gend*endog[i, 4]) # GARCH(1,1) model
        logL = -((-np.log(s) - r ** 2 / s).sum())
        return logL

    def fittedValues(self, endog, est_params, model):
        if(model == "simple"):
            omega, alpha, beta, a, b = est_params
            self.est = a + b * endog[:, 1]
            e = endog[:, 0] - self.est
            n = len(e)
            h = np.ones(n) * 0.01
            h[2] = st.variance(e[0:3])
            for t in range(1, n):
                h[t] = (omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1]))   # GARCH(1,1) model
            # h_sqrt = math.sqrt(h)
            self.z = e * np.sqrt(h)
        elif(model == "seasonal"):
            omega, alpha, beta, a, b, monday, jan, end, gmon, gjan, gend = est_params
            self.est = a + b * endog[:, 1] + monday*endog[:, 2] + jan*endog[:, 3] + end*endog[:, 4]
            e = endog[:, 0] - self.est
            n = len(e)
            h = np.ones(n) * 0.01
            h[2] = st.variance(e[0:3])
            for t in range(1, n):
                h[t] = (omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1])) * math.exp(
                    gmon * endog[t, 2] - gjan * endog[t, 3] - gend * endog[t, 4])  # GARCH(1,1) model
            # h_sqrt = math.sqrt(h)
            self.z = e * np.sqrt(h)
        else:
            omega, alpha, beta, a, b, monday, jan, end, gmon, gjan, gend, gdp = est_params
            self.est = a + b * endog[:, 1] + monday*endog[:, 2] + jan*endog[:, 3] + end*endog[:, 4] + gdp*endog[:,5]
            e = endog[:, 0] - self.est
            n = len(e)
            h = np.ones(n) * 0.01
            h[2] = st.variance(e[0:3])
            for t in range(1, n):
                h[t] = (omega + alpha * e[t - 1] ** 2 + beta * (h[t - 1])) * math.exp(
                    gmon * endog[t, 2] - gjan * endog[t, 3] - gend * endog[t, 4])  # GARCH(1,1) model
            self.z = e * np.sqrt(h)
        return self.est + self.z

    def plotFitted(self, y, simple, seasonal, full):
        matplotlib.style.use('ggplot')

        fitted_val = np.column_stack((y, simple, seasonal, full))
        graph = pd.DataFrame(fitted_val, columns=['actual', 'simple', 'seasonal', 'full'])
        graph.plot(title="Actual vs Estimated Values") #colormap='Blues',
        #g.set_axis_bgcolor('k')
        plt.show()

    def plotFullModel(self, y, full):
        matplotlib.style.use('ggplot')

        fitted_val = np.column_stack((y, full))
        graph = pd.DataFrame(fitted_val, columns=['actual', 'estimated'])
        graph.plot(title="Actual vs Estimated Values")  # colormap='Blues',
        # g.set_axis_bgcolor('k')
        plt.show()