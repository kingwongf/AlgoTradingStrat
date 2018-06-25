
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='MI0zY60sSWgo1OXvwZgo')
import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

data = [go.Scatter(
          x=df.Date,
          y=df['AAPL.Close'])]

py.plotly.iplot(data)

print(df.Date)