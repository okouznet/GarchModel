from biokit.viz import corrplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats.stats import pearsonr

class Test:
    def __init__(self):
        pass

    def CorrelationPlot(self, data):
        corr = data.corr()
        c = corrplot.Corrplot(corr)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        c.plot()
        fig.tight_layout()
        plt.show()

    def CorrelationCoeff(self, a, b):
        return pearsonr(a, b)

    def scatterMatrix(self, data):
        pd.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')

    def testStationary(self, ts):
        # Determing rolling statistics
        rolmean = pd.rolling_mean(ts, window=12)
        rolstd = pd.rolling_std(ts, window=12)
        ts_log = np.log(ts)

        # Plot rolling statistics:
        plt.subplot(211)
        orig = plt.plot(ts, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        # Plot Log (Normalized) Rolling Statistics
        plt.subplot(212)
        moving_avg = pd.rolling_mean(ts_log, 12)
        plt.plot(ts_log)
        plt.plot(moving_avg, color='red')
        plt.title('Normalized Rolling Mean & Standard Deviation')
        plt.show()

        # Perform Dickey-Fuller test:
        print 'Results of Dickey-Fuller Test:'
        dftest = adfuller(ts, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print dfoutput

    def acfTest(self, ts):
        plot_acf(x=ts)
        plt.show()
        lbtest = acorr_ljungbox(x=ts, lags=1, boxpierce=True)
        #lb, lbpval, bp, bppval
        print 'Results of Ljung Box Test and Box Pierce Test: '
        dfoutput = pd.Series(lbtest[0:4],
                             index=['Ljung-Box Test Statistic', 'Ljung-Box p-value: ',
                                    'Box Pierce Test Statistic', 'Box Pierce p-value'])
        print dfoutput

    def residualTest(self, residual):
        plt.subplot(211)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.title('Residuals')
        plt.subplot(212)
        plt.hist(residual, 100, normed=1, facecolor='blue', alpha=0.75)
        plt.legend(loc='best')
        plt.title('Residual Distribution')
        plt.tight_layout()
        plt.show()