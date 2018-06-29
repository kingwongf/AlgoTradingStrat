import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.nan)

import plotly as py

py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd

fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col="Dates")
macronews = pd.read_csv("../PData/New_Duplicates_Removed_MacroNews09_11_2017_11_06_2018.csv", header=0,
                        index_col="DateTime")

# fx_news_concatenate = pd.concat([macronews,fx_data],axis=1,join='inner')


fx_news_concatenate = fx_data.join(macronews)

print(fx_news_concatenate.head())

USDGBP_ratio = np.divide(np.multiply(fx_news_concatenate['USDEUR_Close_Ask'], fx_news_concatenate['EURGBP_Close_Ask']),
                         fx_news_concatenate['USDGBP_Close_Ask'])

USDGBP_ratio = go.Scattergl(x=fx_data.index, y=USDGBP_ratio, name="Implied Cross / Market Cross"
                            , line=dict(color='#cf1717'), opacity=0.8)

mkt_vol = fx_news_concatenate['Volatility']
print(np.sum(mkt_vol))
macro_news_impact = go.Scattergl(x=fx_data.index, y=mkt_vol, name="mmacro news impact", yaxis='y2', mode='markers')

transit_prob_P00 = go.Scattergl(
    x=fx_data.index[fx_data.index.get_loc('1/1/2018 0:00'):fx_data.index.get_loc('11/6/2018 19:35')],
    y=transit_matrix['P00'])
data = [USDGBP_ratio, macro_news_impact]

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

fig = go.Figure(data=data, layout=layout)
plot_url = py.plotly.plot(fig, filename='multiple-axes-double-newone')
