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
    response = input("Do you need to collect data (y/n): ")
    if response == 'y':
        stock_data = d.prepocess_stock_data(stocks=['xlp', 'xly', 'xle', 'xlk', 'xlf', 'xlv', 'xli', 'xlb', 'xlre', 'xlu'])
        shock_data = d.preprocess_macroeconomic_data()
    else:
        stock_data = pd.read_csv('data/stocks.csv')
        shock_data = pd.read_csv('data/shocks.csv')

    data = pd.merge(stock_data, shock_data, how='outer', on='Dates')

    #basic statistics test
    test = tests.Test()
    basic_tests = input("Do you want to run basic statistical tests (y/n)?: ")
    if(basic_tests == 'y'):
        test.stationary_test(ts=stock_data['xlp'][1000:1200])
        df = np.column_stack((data['xlp'], data['GDP'], data['UMCSENT'], data['CPIAUCSL']))
        dftest = pd.DataFrame(df, columns=['xlp', 'GDP', 'UMCSENT', 'CPI'])
        test.correlation_plot(data=dftest)
        test.scatter_matrix(data=dftest)

    m = model.Model()
    ylag = data['xlp'].shift(periods=1).values.astype(float)
    y = data['xlp'].values.astype(float)
    gdp = data['GDP'].fillna(0)

    y = y[2:2370]
    ylag = ylag[2:2370]
    gdp = gdp[2:2370]

    seasonality_data = d.get_seasonality_effects(data=data['Dates'])
    simple_data = np.column_stack((y, ylag))
    seasonal_data = np.column_stack((y, ylag, seasonality_data[2:2370, 0], seasonality_data[2:2370, 1], seasonality_data[2:2370, 2]))
    full_data = np.column_stack((y, ylag, seasonality_data[2:2370, 0], seasonality_data[2:2370, 1], seasonality_data[2:2370, 2], gdp))

    #run MLE estimation
    simple = optimize.fmin(m.garch11_model, np.array([.5, .5, .5, .5, .5]), args=(simple_data,), maxfun=100000, maxiter=10000, full_output=1)
    seasonal = optimize.fmin(m.garch11_controlled_model, np.array([.5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0]), args=(seasonal_data,), maxfun=100000, maxiter=10000, full_output=1)
    full = optimize.fmin(m.garch11_full_model, np.array([.5, .5, .5, .5, .5, 0, 0, 0, 0, 0, 0, 0]), args=(full_data,), maxfun=100000, maxiter=10000, full_output=1)

    #get estimated parameters
    simple_est = np.abs(simple[0])
    seasonal_est = np.abs(seasonal[0])
    full_est = np.abs(full[0])

    #calculate fitted values
    simple_fitted = m.fitted_values(endog=simple_data, est_params=simple_est, model="simple")
    seasonal_fitted = m.fitted_values(endog=seasonal_data, est_params=seasonal_est, model="seasonal")
    full_fitted = m.fitted_values(endog=full_data, est_params=full_est, model="full")

    m.plot_fitted_values(y=y[1000:1050], simple=simple_fitted[1000:1050], seasonal=seasonal_fitted[1000:1050], full=full_fitted[1000:1050])
    m.plot_full_model(y=y[1000:1050], full=full_fitted[1000:1050])

    #residual tests
    residual_tests = input("Do you want to run residual distribution tests (y/n)?: ")
    if residual_tests == 'y':
        residuals = y - full_fitted
        test.residual_test(residual=residuals[1000:1200])
        test.acf_test(ts=residuals[1000:1200])
        test.t_test(a=simple_fitted, b=full_fitted)
