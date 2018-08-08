import pandas as pd
import deepdish as dd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')




############


## read list of optimal models
from for_fx_find_opt_model import find_opt

opt_model = find_opt()      ## list of str(name) of optimal models


## read transition matrix
# transit_global =  dict.fromkeys(opt_model, 0)
transit_global = []
for name in opt_model:
    print(name)
    store = pd.HDFStore(name)
    df = pd.read_hdf(store, name[28:-3])    ## drop name of the directory to ".h5"
    store.close()
    transition_matrix = df.drop(labels='Score', axis=1)
    # print(transition_matrix)
    list_transit = transition_matrix.T.values.tolist()
    # print(list_transit[1])
    # print(list_transit)
    transit_global.append(list_transit)     ## list in the order of find_opt_model

#### read bid, ask price
fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")

#####  run backtest model
from algo_trade_backtest import main
Net_pos_global = {}
shift_global = {}
for ind_global, transit_m in enumerate(transit_global):
    label = opt_model[ind_global][31:-3]
    try:
        ticker = label[label.index('USD'):label.index('USD')+6]
    except:
        ticker = label[label.index('EUR'):label.index('EUR')+6]

    ask = fx_data[ticker + '_Close_Ask'][15000:15000+6048]
    bid = fx_data[ticker + '_Close_Bid'][15000:15000+6048]

    ## if xbt: use xbtfx, if xetusd/eur: use cxfx

    P_label_list = ['P' + str(i) + str(j) for i in range(0, len(transit_m)) for j in range(0, len(transit_m))]

    for ind, P in enumerate(transit_m):
        Net_pos, shift = main(P=P[:6048], bid=bid, ask=ask)
        Net_pos_global[label + "_" + P_label_list[ind]] = Net_pos
        shift_global[label + "_" + P_label_list[ind]] = shift


# print(transit_global['bidask_spd_XBTUSD_10'])

dd.io.save('../PData/FX_Net_pos_global.h5', Net_pos_global, compression=None)
dd.io.save('../PData/FX_Shift_global.h5',shift_global, compression=None)


#############
## read bid ask price
# fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
# xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")[15000:15000+6048]
# xbtfx_data = fx_data.join(xbt_data)
# xbtfx_data = xbtfx_data.dropna()




# #########
#
# ## read transition matrix
#
# store = pd.HDFStore("../PData/Crypto_transit_matrix_/6048_backtesting/bidask_spd_XBTUSD_7.h5")
# df = pd.read_hdf(store, "bidask_spd_XBTUSD_7")
# store.close()
#
#
# ## read return
#
# fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
# xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
# cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")
#
#
# xbtfx_data = fx_data.join(xbt_data)
# cxfx_data = fx_data.join(cx_data_0208)
#
# xbtfx_data = xbtfx_data.dropna()
# cxfx_data = cxfx_data.dropna()
#
#
# ret_XBTUSD_ask = np.diff(np.log(cxfx_data['XBTUSD_Close_Ask']))
#
# XBTUSD_ask = cxfx_data['XBTUSD_Close_Ask']
#
# df = df.drop(labels='Score', axis=1)
#
#
# print(len(XBTUSD_ask.tolist()[15000:15000+6048]))
# print(len(df["P00"]))
# df['XBTUSD_ask'] = XBTUSD_ask.tolist()[15000:15000+6048]
#
#
# # fig = plt.figure()
# fig = df.plot(subplots=True)
#
# plot_url = py.plotly.plot_mpl(fig)
# plt.show()
#
