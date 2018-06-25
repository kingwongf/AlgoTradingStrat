import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("../PData/USDGBP_Rato_Transit_Matrix_09012018_09032018.csv", header = 0 )

P01 = go.Scatter(x=df["Dates"],y=df['P01'], name = "P01", line = dict(color = '#17BECF'), opacity = 0.8)
P11 = go.Scatter(x=df["Dates"],y=df['P11'], name = "P11", line = dict(color = '#cf5a17'), opacity = 0.8)
P10 = go.Scatter(x=df["Dates"],y=df['P10'], name = "P10", line = dict(color = '#17cf5a'), opacity = 0.8)
P00 = go.Scatter(x=df["Dates"],y=df['P00'], name = "P00", line = dict(color = '#cf1717'), opacity = 0.8)
data = [P01,P11,P10,P00]


layout = go.Layout(title='Transit Matrix of USDGBP Implied to Market Cross Rate Ratio 09012018 09032018',
        yaxis=dict(title='Transition Probabilities'))




fig = dict(data=data, layout = layout)
py.plotly.iplot(fig, filename = "Transit Matrix of USDGBP Implied to Market Cross Rate Ratio 09012018 09032018")

