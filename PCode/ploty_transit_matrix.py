import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd

df_transit_matrix = pd.read_csv("../PData/transitr_matrix.csv", header =0)

df_macro_news_impact = pd.read_csv("../PData/New_Duplicates_Removed_MacroNews09_11_2017_11_06_2018_EURUSD.csv"
                                   , header=0)
train_period = df_macro_news_impact.index[df_macro_news_impact.get_loc('1/1/2018 0:00')
                                          :df_macro_news_impact.get_loc('11/6/2018 19:35')]

P01 = go.Scatter(x=train_period,y=df_transit_matrix['P01'], name = "P01", line = dict(color = '#17BECF'), opacity = 0.8)
P11 = go.Scatter(x=train_period,y=df_transit_matrix['P11'], name = "P11", line = dict(color = '#cf5a17'), opacity = 0.8)
P10 = go.Scatter(x=train_period,y=df_transit_matrix['P10'], name = "P10", line = dict(color = '#17cf5a'), opacity = 0.8)
P00 = go.Scatter(x=train_period,y=df_transit_matrix['P00'], name = "P00", line = dict(color = '#cf1717'), opacity = 0.8)

macronews = go.Scatter(x=train_period, y= df_macro_news_impact["Volatility"], mode="markers")

data = [P01,P11,P10,P00, macronews]


layout = go.Layout(title='Transit Matrix of USDGBP Implied to Market Cross Rate Ratio 1/1/2018 0:00 to 11/6/2018 19:35',
        yaxis=dict(title='Transition Probabilities'))




fig = dict(data=data, layout = layout)
py.plotly.iplot(fig, filename = "Transit Matrix of USDGBP Implied to Market Cross Rate Ratio 09012018 09032018")

