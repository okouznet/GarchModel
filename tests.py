from biokit.viz import corrplot
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import ProbPlot
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_ind


class Test:
    def __init__(self):
        pass

    def correlation_plot(self, data):
        corr = data.corr()
        c = corrplot.Corrplot(corr)
        fig = plt.figure()
        c.plot()
        fig.tight_layout()
        plt.show()

    def correlation_coefficients(self, a, b):
        return pearsonr(a, b)

    def scatter_matrix(self, data):
        pd.scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')

    def stationary_test(self, ts):

        # Determing rolling statistics
        rolmean = pd.rolling_mean(ts, window=12)
        rolstd = pd.rolling_std(ts, window=12)

        # Plot rolling statistics:
        plt.style.use('ggplot')
        plt.subplot(111)
        plt.plot(ts, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(ts, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def acf_test(self, ts):
        plot_acf(x=ts)
        plt.show()
        lbtest = acorr_ljungbox(x=ts, lags=1, boxpierce=True)
        # lb, lbpval, bp, bppval
        print('Results of Ljung Box Test and Box Pierce Test: ')
        dfoutput = pd.Series(lbtest[0:4],
                             index=['Ljung-Box Test Statistic', 'Ljung-Box p-value: ',
                                    'Box Pierce Test Statistic', 'Box Pierce p-value'])
        print(dfoutput)

    def residual_test(self, residual):
        plt.subplot(211)
        plt.plot(residual, 'bo', label='Residuals')
        plt.legend(loc='best')
        plt.title('Residuals')
        plt.subplot(212)
        plt.hist(residual, 100, normed=1, facecolor='blue', alpha=0.75)
        plt.legend(loc='best')
        plt.title('Residual Distribution')
        plt.tight_layout()
        plt.show()

        probplot = ProbPlot(residual)
        probplot.qqplot()

    def t_test(self, a, b):
        x = ttest_ind(a=a, b=b)
        print(x)
