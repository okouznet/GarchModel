from biokit.viz import corrplot
import matplotlib.pyplot as plt
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


   # def ACFTest(self, a, b):
