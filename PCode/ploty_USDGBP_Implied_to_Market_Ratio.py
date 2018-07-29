import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd
from plotly import tools


fx_data = pd.read_csv("/Users/kingf.wong/Documents/GitHub/AlgoTradingStrat/PData/FX_PData.csv", header=0, index_col ="Dates")


USDGBP_ratio = np.multiply(fx_data['USDEUR_Close_Ask'],fx_data['EURGBP_Close_Ask']) - fx_data['USDGBP_Close_Ask']

USDGBP_bidask = fx_data['USDGBP_Close_Ask'] - fx_data['USDGBP_Close_Bid']

USDGBP_mid  = (fx_data['USDGBP_Close_Ask'] + fx_data['USDGBP_Close_Bid'])/2


USDGBP_arb_profit = go.Scatter(x=fx_data.index, y=USDGBP_ratio, name = "Implied Cross - Market Cross"
                          , line = dict(color = '#cf1717'), opacity = 0.8)
USDGBP_bidask_spd = go.Scatter(x=fx_data.index, y = USDGBP_bidask, name = "Bid Ask Spread"
                          , line = dict(color = '#0000ff'), opacity = 0.8)

USDGBP_ret = go.Scatter(x=fx_data.index, y = np.diff(np.log(fx_data['USDGBP_Close_Ask'])), name = "USDGBP Price Return"
                          , line = dict(color = '#008000'), opacity = 0.8)

USDGBP_mid_plot = go.Scatter(x=fx_data.index, y = USDGBP_mid[1::], name = "USDGBP Mid Price", opacity = 0.8)

# USDGBP_OHLC = go.Ohlc(x=fx_data.index,
#                 open=fx_data['USDGBP_Open_Ask'],
#                 high=fx_data['USDGBP_High_Ask'],
#                 low=fx_data['USDGBP_Low_Ask'],
#                 close=fx_data['USDGBP_Close_Ask'])

# data = [USDGBP_arb_profit, USDGBP_bidask_spd, USDGBP_ret, USDGBP_mid]

layout = go.Layout(title='USDGBP')

fig = tools.make_subplots(rows=4, cols=1,shared_xaxes=True)

fig.append_trace(USDGBP_mid_plot, 1, 1)
fig.append_trace(USDGBP_ret, 2, 1)
fig.append_trace(USDGBP_arb_profit, 3, 1)
fig.append_trace(USDGBP_bidask_spd, 4, 1)



# fig = dict(data=data, layout = layout)

py.plotly.iplot(fig, filename = "USDGBP")


# fig = tools.make_subplots(rows=3, cols=1)
#
# fig.append_trace(USDGBP_arb_profit, 1, 1)
# fig.append_trace(USDGBP_bidask_spd, 2, 1)
# fig.append_trace(USDGBP_OHLC, 3, 1)
#
# fig['layout'].update(height=600, width=800, title='USDGBP feature')
# py.plotly.iplot(fig, filename='USDGBP feature')