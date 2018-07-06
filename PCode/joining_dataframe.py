import numpy as np
np.set_printoptions(threshold=np.nan)

import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd


fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")
macronews = pd.read_csv("../PData/Dates_Corrected_Duplicates_Removed_MacroNews09_11_2017_11_06_2018.csv"
                        , header=0, index_col ="DateTime")
transit_matrix = pd.read_csv("../PData/transit_matrix.csv", header=0)

print(np.sum(macronews["Volatility"]))
# fx_news_data_join = pd.concat([macronews,fx_data],axis=1,join='inner')


fx_news_data_join = fx_data.join(macronews)
fx_news_data_join.to_csv("fx_news_impact_concatenate.csv")
print(np.sum(fx_news_data_join['Volatility']))

USDGBP_ratio = np.divide(np.multiply(fx_news_data_join['USDEUR_Close_Ask'],fx_news_data_join['EURGBP_Close_Ask']), fx_news_data_join['USDGBP_Close_Ask'])


USDGBP_ratio_time_plot = go.Scattergl(x=fx_news_data_join.index, y=USDGBP_ratio, name = "Implied Cross / Market Cross"
                           , line = dict(color = '#cf1717'), opacity = 0.8)

mkt_vol = fx_news_data_join['Volatility']

macro_news_impact = go.Scattergl(x=fx_news_data_join.index, y=mkt_vol, name = "mmacro news impact",yaxis='y2', line = dict(color = '#17cf17'))

transit_prob_P00 = go.Scattergl(x=fx_data.index[fx_data.index.get_loc('1/1/2018 0:00'):fx_data.index.get_loc('11/6/2018 19:35')], y = transit_matrix['P00'])
transit_prob_P01 = go.Scattergl(x=fx_data.index[fx_data.index.get_loc('1/1/2018 0:00'):fx_data.index.get_loc('11/6/2018 19:35')], y = transit_matrix['P01'])
transit_prob_P10 = go.Scattergl(x=fx_data.index[fx_data.index.get_loc('1/1/2018 0:00'):fx_data.index.get_loc('11/6/2018 19:35')], y = transit_matrix['P10'])
transit_prob_P11 = go.Scattergl(x=fx_data.index[fx_data.index.get_loc('1/1/2018 0:00'):fx_data.index.get_loc('11/6/2018 19:35')], y = transit_matrix['P11'])


data = [USDGBP_ratio_time_plot, macro_news_impact]

layout = go.Layout(
    title='Double Y Axis Example',
    yaxis=dict(
        title='yaxis title'
    ),
    yaxis2=dict(
        title='yaxis2 title',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)

# fig = go.Figure(data=data, layout=layout)
# plot_url = py.plotly.plot(fig, filename='USDGBP_new_impact')

line_reg = go.Scatter(x = mkt_vol, y=np.diff(USDGBP_ratio), mode= 'markers')
data = [line_reg]
fig = go.Figure(data=data)
plot_url = py.plotly.plot(fig, filename = 'USDGBP_news_impact_Line_Reg')
