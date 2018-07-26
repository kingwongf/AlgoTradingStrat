import pandas as pd

import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go



## read transition matrix

store = pd.HDFStore("../PData/Crypto_transit_matrix_/6048_backtesting/bidask_spd_XBTUSD_7.h5")
df = pd.read_hdf(store, "bidask_spd_XBTUSD_7")
store.close()


## read return

fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")


xbtfx_data = fx_data.join(xbt_data)
cxfx_data = fx_data.join(cx_data_0208)

xbtfx_data = xbtfx_data.dropna()
cxfx_data = cxfx_data.dropna()


ret_XBTUSD_ask = np.diff(np.log(cxfx_data['XBTUSD_Close_Ask']))

XBTUSD_ask = cxfx_data['XBTUSD_Close_Ask'].tolist()



print(df.columns.values[0])

data = []
for column in df:
    data.append(go.Scatter(
        x=len(df),  # assign x as the dataframe column 'x'
        y=df[column],
        name= column
    ))

data.append(XBTUSD_ask)

print(data)
layout = go.Layout(title='XBTUSD')
fig = dict(data=data, layout = layout)

py.plotly.iplot(fig, filename = "Transit Matrix of XBTUSD Implied to Market Cross Rate")


