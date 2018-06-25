import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd


fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")


USDGBP_ratio = np.divide(np.multiply(fx_data['USDEUR_Close_Ask'],fx_data['EURGBP_Close_Ask']), fx_data['USDGBP_Close_Ask'])


USDGBP_ratio = go.Scatter(x=fx_data.index, y=USDGBP_ratio, name = "Implied Cross / Market Cross"
                          , line = dict(color = '#cf1717'), opacity = 0.8)
data = [USDGBP_ratio]


layout = go.Layout(title='Implied USDGBP Cross to Market USDGBP Cross Ratio')


fig = dict(data=data, layout = layout)
py.plotly.iplot(fig, filename = "Implied USDGBP Cross to Market USDGBP Cross Ratio")


