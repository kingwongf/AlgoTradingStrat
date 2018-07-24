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


## read data
fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")


cxfx_data = fx_data.join(cx_data_0208)
cxfx_data = cxfx_data.dropna()


### price return


## XBT
ret_XBTUSD_ask = np.diff(cxfx_data['XBTUSD_Close_Ask'])

ret_XBTUSD_bid = np.diff(cxfx_data['XBTUSD_Close_Bid'])

ret_XBTEUR_ask = np.diff(cxfx_data['XBTEUR_Close_Ask'])

ret_XBTEUR_bid = np.diff(cxfx_data['XBTEUR_Close_Bid'])


##  XET
ret_XETUSD_ask = np.diff(cxfx_data['XETUSD_Close_Ask'])

ret_XETUSD_bid = np.diff(cxfx_data['XETUSD_Close_Bid'])

ret_XETEUR_ask = np.diff(cxfx_data['XETEUR_Close_Ask'])

ret_XETEUR_bid = np.diff(cxfx_data['XETEUR_Close_Bid'])

## XRP

ret_XRPUSD_ask = np.diff(cxfx_data['XRPUSD_Close_Ask'])

ret_XRPUSD_bid = np.diff(cxfx_data['XRPUSD_Close_Bid'])


### bidask spd


## XBT

bidask_spd_XBTUSD = cxfx_data['XBTUSD_Close_Ask'] - cxfx_data['XBTUSD_Close_Bid']

bidask_spd_XBTEUR = cxfx_data['XBTEUR_Close_Ask'] - cxfx_data['XBTEUR_Close_Bid']

## XET

bidask_spd_XETUSD = cxfx_data['XETUSD_Close_Ask'] - cxfx_data['XETUSD_Close_Bid']

bidask_spd_XETEUR = cxfx_data['XETEUR_Close_Ask'] - cxfx_data['XETEUR_Close_Bid']


## XRP

bidask_spd_XRPUSD = cxfx_data['XRPUSD_Close_Ask'] - cxfx_data['XRPUSD_Close_Bid']


## arb_proft

arb_profit_XBTUSD_ask = np.multiply(1/cxfx_data['USDEUR_Close_Bid'], cxfx_data['XBTEUR_Close_Ask']) - \
                        cxfx_data['XBTUSD_Close_Ask']

arb_profit_XBTUSD_bid = np.multiply(1/cxfx_data['USDEUR_Close_Ask'], cxfx_data['XBTEUR_Close_Bid']) - \
                        cxfx_data['XBTUSD_Close_Bid']
###############

arb_profit_XBTEUR_ask = np.multiply(cxfx_data['USDEUR_Close_Ask'], cxfx_data['XBTUSD_Close_Ask']) - \
                        cxfx_data['XBTEUR_Close_Ask']

arb_profit_XBTEUR_bid = np.multiply(cxfx_data['USDEUR_Close_Bid'], cxfx_data['XBTUSD_Close_Bid']) - \
                        cxfx_data['XBTEUR_Close_Bid']
###############

arb_profit_XETUSD_ask = np.multiply(1/cxfx_data['USDEUR_Close_Bid'], cxfx_data['XETEUR_Close_Ask']) - \
                        cxfx_data['XETUSD_Close_Ask']

arb_profit_XETUSD_bid = np.multiply(1/cxfx_data['USDEUR_Close_Ask'], cxfx_data['XETEUR_Close_Bid']) - \
                        cxfx_data['XETUSD_Close_Bid']
###############

arb_profit_XETEUR_ask = np.multiply(cxfx_data['USDEUR_Close_Ask'], cxfx_data['XETUSD_Close_Ask']) - \
                        cxfx_data['XETEUR_Close_Ask']

arb_profit_XETEUR_bid = np.multiply(1/cxfx_data['USDEUR_Close_Ask'], cxfx_data['XETUSD_Close_Bid']) - \
                        cxfx_data['XETEUR_Close_Bid']



def features_analysis(arb, bidask, ret):

    def OLS(y,x):
        x = x.reshape(-1,1)
        y = y.reshape(-1, 1)
        linreg = LinearRegression(fit_intercept=True)
        linreg.fit(x,y)
        linreg.get_params()
        return [np.round(linreg.score(x,y),decimals=2), np.round(linreg.coef_[0][0],decimals=2)]

    arb_ret = OLS(np.diff(arb), ret)
    bidask_ret = OLS(np.diff(bidask), ret)
    arb_bidask = OLS(np.diff(arb), np.diff(bidask))
    return [arb_ret, bidask_ret, arb_bidask]


# ret_list = []

ret_list = [ret_XBTUSD_ask, ret_XBTUSD_bid, ret_XBTEUR_ask, ret_XBTEUR_bid, ret_XETUSD_ask, ret_XETUSD_bid,
            ret_XETEUR_ask, ret_XETEUR_bid]

ret_list_label = ["ret_XBTUSD_ask", "ret_XBTUSD_bid", "ret_XBTEUR_ask", "ret_XBTEUR_bid",
                  "ret_XETUSD_ask", "ret_XETUSD_bid", "ret_XETEUR_ask", "ret_XETEUR_bid"]

# bidask_list = [bidask_spd_USDEUR, bidask_spd_USDGBP, bidask_spd_USDJPY, bidask_spd_EURJPY, bidask_spd_EURGBP]

bidask_list = [bidask_spd_XBTUSD, bidask_spd_XBTUSD, bidask_spd_USDJPY, bidask_spd_USDJPY,
               bidask_spd_USDEUR, bidask_spd_USDEUR]


arb_list = [arb_profit_XBTUSD_ask, arb_profit_XBTUSD_bid, arb_profit_XETUSD_ask, arb_profit_XETUSD_bid]


df = [features_analysis(arb_list[i],bidask_list[i],ret_list[i]) for i in range(len(arb_list))]


df = pd.DataFrame(df, index=ret_list_label, columns=["arb / ret", "bidask/ ret", "arb/ bidask"])

# [print(i,j,k) for i in arb_list_label for j in bidask_list_label for k in ret_list_label]

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