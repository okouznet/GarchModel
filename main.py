import model
import data
import tests
import pandas as pd
import numpy as np
from scipy import optimize


if __name__ == '__main__':
    d = data.Data()
    stock_data = pd.DataFrame()
    shock_data = pd.DataFrame()
    response = raw_input("Do you need to collect data (y/n): ")
    if(response == 'y'):
        print("test")
        stock_data = d.stockData(stocks=['xlp','xly', 'xle', 'xlk', 'xlf', 'xlv', 'xli', 'xlb', 'xlre', 'xlu'])
        shock_data = d.shockData()
    else:
        stock_data = pd.read_csv('stocks.csv')
        shock_data = pd.read_csv('shocks.csv')

    data = pd.merge(stock_data, shock_data, how='outer', on='Dates')

    #basic statistics test
    test = tests.Test()
    basic_tests = raw_input("Do you want to run basic statistical tests (y/n)?: ")
    if(basic_tests == 'y'):
        test.testStationary(ts=stock_data['xlp'][1000:1200])
        df = np.column_stack((data['xlp'], data['GDP'], data['UMCSENT'], data['CPIAUCSL']))
        dftest = pd.DataFrame(df, columns=['xlp', 'GDP', 'UMCSENT', 'CPI'])
        test.CorrelationPlot(data=dftest)
        test.scatterMatrix(data=dftest)

    m = model.Model()
    ylag = data['xlp'].shift(periods=1).values.astype(float)
    y = data['xlp'].values.astype(float)
    gdp = data['GDP'].fillna(0)

    y = y[2:2370]
    ylag = ylag[2:2370]
    gdp = gdp[2:2370]

    seasonality_data = d.getSeasonalityEffects(data=data['Dates'])
    simple_data = np.column_stack((y, ylag))
    seasonal_data = np.column_stack((y, ylag, seasonality_data[2:2370, 0], seasonality_data[2:2370, 1], seasonality_data[2:2370, 2]))
    full_data = np.column_stack((y, ylag, seasonality_data[2:2370, 0], seasonality_data[2:2370, 1], seasonality_data[2:2370, 2], gdp))

    #run MLE estimation
    simple = optimize.fmin(m.GARCH11_logLSimple,np.array([.5, .5, .5, .5, .5]), args=(simple_data,), maxfun=100000, maxiter=10000, full_output=1)
    seasonal = optimize.fmin(m.GARCH11_logLSeasonal,np.array([.5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0]), args=(seasonal_data,), maxfun=100000, maxiter=10000, full_output=1)
    full = optimize.fmin(m.GARCH11_logLFull,np.array([.5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0]), args=(full_data,), maxfun=100000, maxiter=10000, full_output=1)

    #get estimated parameters
    simple_est = np.abs(simple[0])
    seasonal_est = np.abs(seasonal[0])
    full_est = np.abs(full[0])

    #calculate fitted values
    simple_fitted = m.fittedValues(endog=simple_data, est_params=simple_est, model="simple")
    seasonal_fitted = m.fittedValues(endog=seasonal_data, est_params=seasonal_est, model="seasonal")
    full_fitted = m.fittedValues(endog=full_data, est_params=full_est, model="full")

    m.plotFitted(y=y[1000:1050], simple=simple_fitted[1000:1050], seasonal=seasonal_fitted[1000:1050], full=full_fitted[1000:1050])
    m.plotFullModel(y=y[1000:1050], full=full_fitted[1000:1050])

    #residual tests
    resid_tests = raw_input("Do you want to run residual distribution tests (y/n)?: ")
    if(resid_tests == 'y'):
        resid = y - full_fitted
        test.residualTest(residual=resid[1000:1200])
        test.acfTest(ts=resid[1000:1200])
        test.ttest(a=simple_fitted, b=full_fitted)
