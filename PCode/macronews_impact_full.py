import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import pandas as pd


fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")
macronews = pd.read_csv("../PData/MacroNews09_11_2017_11_06_2018.csv", header=0, index_col ="DateTime")

fx_join = fx_data.join(macronews)


USDGBP_ratio = np.divide(np.multiply(fx_data['USDEUR_Close_Ask'],fx_data['EURGBP_Close_Ask']), fx_data['USDGBP_Close_Ask'])

usdgbp_ratio = go.Scattergl(x= fx_data.index, y= USDGBP_ratio)

macronews_impact = go.Scatter(x= macronews.index, y= macronews["Volatility"], mode="markers")

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
data = [usdgbp_ratio, macronews_impact]
fig = go.Figure(data=data, layout=layout)
plot_url = py.plotly.plot(fig, filename='Ratio Macronews Impact')