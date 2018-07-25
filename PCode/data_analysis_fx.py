import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd
from plotly import tools
import matplotlib as plt
plt.use('TkAgg')
import statsmodels.api as stats
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats
import numpy as np


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

## read data


fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")




ret_USDGBP_ask = np.diff(np.log(fx_data['USDGBP_Close_Ask']))

ret_USDGBP_bid = np.diff(np.log(fx_data['USDGBP_Close_Bid']))

ret_USDJPY_ask = np.diff(np.log(fx_data['USDJPY_Close_Ask']))

ret_USDJPY_bid = np.diff(np.log(fx_data['USDJPY_Close_Bid']))

ret_USDEUR_ask = np.diff(np.log(fx_data['USDEUR_Close_Ask']))

ret_USDEUR_bid = np.diff(np.log(fx_data['USDEUR_Close_Bid']))

ret_EURJPY_ask = np.diff(np.log(fx_data['EURJPY_Close_Ask']))

ret_EURJPY_bid = np.diff(np.log(fx_data['EURJPY_Close_Bid']))

ret_EURGBP_ask = np.diff(np.log(fx_data['EURGBP_Close_Ask']))

ret_EURGBP_bid = np.diff(np.log(fx_data['EURGBP_Close_Bid']))



bidask_spd_USDEUR = fx_data['USDEUR_Close_Ask'] - fx_data['USDEUR_Close_Bid']

bidask_spd_USDGBP = fx_data['USDGBP_Close_Ask'] - fx_data['USDGBP_Close_Bid']

bidask_spd_USDJPY = fx_data['USDJPY_Close_Ask'] - fx_data['USDJPY_Close_Bid']

bidask_spd_EURJPY = fx_data['EURJPY_Close_Ask'] - fx_data['EURJPY_Close_Bid']

bidask_spd_EURGBP = fx_data['EURGBP_Close_Ask'] - fx_data['EURGBP_Close_Bid']


arb_profit_USDGBP_ask = np.multiply(fx_data['USDEUR_Close_Ask'], fx_data['EURGBP_Close_Ask']) - \
                        fx_data['USDGBP_Close_Ask']

arb_profit_USDGBP_bid = np.multiply(fx_data['USDEUR_Close_Bid'], fx_data['EURGBP_Close_Bid']) - \
                        fx_data['USDGBP_Close_Bid']

arb_profit_USDJPY_ask = np.multiply(fx_data['USDEUR_Close_Ask'], fx_data['EURJPY_Close_Ask']) - \
                        fx_data['USDJPY_Close_Ask']

arb_profit_USDJPY_bid = np.multiply(fx_data['USDEUR_Close_Bid'], fx_data['EURJPY_Close_Bid']) - \
                        fx_data['USDJPY_Close_Bid']

arb_profit_USDEUR_ask = np.multiply(np.divide(1,fx_data['EURJPY_Close_Bid']), fx_data['USDJPY_Close_Ask']) - \
                        fx_data['USDJPY_Close_Ask']

arb_profit_USDEUR_bid = np.multiply(np.divide(1,fx_data['EURJPY_Close_Ask']), fx_data['USDJPY_Close_Bid']) - \
                        fx_data['USDJPY_Close_Bid']

## correlation scatter plot
# USDGBP_arb_profit_ret = go.Scatter(x=np.diff(USDGBP_arb[:-1]), y = USDGBP_ret, mode = 'markers',
#                                    name = "Implied Cross - Market Cross"
#                           , line = dict(color = '#cf1717'), opacity = 0.8)
# USDGBP_bidask_spd_ret = go.Scatter(x=np.diff(USDGBP_bidask), y = USDGBP_ret, mode = 'markers', name = "Bid Ask Spread"
#                           , line = dict(color = '#0000ff'), opacity = 0.8)
#
# USDGBP_arb_spd = go.Scatter(x=USDGBP_bidask, y = USDGBP_arb, mode = 'markers', name = "Bid Ask Spread"
#                           , line = dict(color = '#0000ff'), opacity = 0.8)

def features_analysis(arb, bidask, ret):

    def OLS(y,x):
        x = x.reshape(-1,1)
        y = y.reshape(-1, 1)
        linreg = LinearRegression(fit_intercept=True)
        linreg.fit(x,y)
        return [np.round(linreg.score(x,y),decimals=2), np.round(linreg.p[0][0],decimals=2)]

    arb_ret = OLS(np.diff(arb), ret)
    bidask_ret = OLS(np.diff(bidask), ret)
    arb_bidask = OLS(np.diff(arb), np.diff(bidask))
    return [arb_ret, bidask_ret, arb_bidask]


# ret_list = [ret_USDGBP_ask, ret_USDGBP_bid, ret_USDJPY_ask, ret_USDJPY_bid, ret_USDEUR_ask, ret_USDEUR_bid,
#             ret_EURJPY_ask, ret_EURJPY_bid, ret_EURGBP_ask, ret_EURGBP_bid]

ret_list = [ret_USDGBP_ask, ret_USDGBP_bid, ret_USDJPY_ask, ret_USDJPY_bid, ret_USDEUR_ask, ret_USDEUR_bid]


ret_list_label = ["USDGBP ask", "USDGBP bid", "USDJPY ask", "USDJPY bid",
                  "USDEUR ask", "USDEUR bid"]

# bidask_list = [bidask_spd_USDEUR, bidask_spd_USDGBP, bidask_spd_USDJPY, bidask_spd_EURJPY, bidask_spd_EURGBP]

bidask_list = [bidask_spd_USDGBP, bidask_spd_USDGBP, bidask_spd_USDJPY, bidask_spd_USDJPY,
               bidask_spd_USDEUR, bidask_spd_USDEUR]


arb_list = [arb_profit_USDGBP_ask, arb_profit_USDGBP_bid, arb_profit_USDJPY_ask, arb_profit_USDJPY_bid,
            arb_profit_USDEUR_ask, arb_profit_USDEUR_bid]


df = [features_analysis(arb_list[i],bidask_list[i],ret_list[i]) for i in range(len(arb_list))]


df = pd.DataFrame(df, index=ret_list_label, columns=["arb / ret", "bidask/ ret", "arb/ bidask"])

# [print(i,j,k) for i in arb_list_label for j in bidask_list_label for k in ret_list_label]

print(df)

df.to_csv("../PData/fx_data_analysis.csv")








## graphing/ plotting

# data = [USDGBP_arb_profit_ret, USDGBP_bidask_spd_ret]
#
#
# layout = go.Layout(title='Correlations with USDGBP Price returns')
#
#
# fig = tools.make_subplots(rows=1, cols=3)
#
#
# fig.append_trace(USDGBP_bidask_spd_ret, 1, 1)
# fig.append_trace(USDGBP_arb_profit_ret, 1, 2)
# fig.append_trace(USDGBP_arb_spd, 1, 3)
#
# fig['layout'].update(height=600, width=800, title= 'Correlations with USDGBP Price returns')
# py.plotly.iplot(fig, filename= 'Correlations with USDGBP Price returns')






# fig = dict(data=data, layout = layout)
# py.plotly.iplot(fig, filename = "USDGBP time series")


# fig = tools.make_subplots(rows=3, cols=1)
#
# fig.append_trace(USDGBP_arb_profit, 1, 1)
# fig.append_trace(USDGBP_bidask_spd, 2, 1)
# fig.append_trace(USDGBP_OHLC, 3, 1)
#
# fig['layout'].update(height=600, width=800, title='USDGBP feature')
# py.plotly.iplot(fig, filename='USDGBP feature')