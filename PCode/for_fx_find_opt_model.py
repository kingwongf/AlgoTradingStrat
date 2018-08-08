import matplotlib as mpl
mpl.use('TkAgg')
import pandas as pd
import numpy as np


def find_opt():
    def BIC(df, n_states):
        T = 15000
        p = n_states**2 + 2*n_states -1
        df['BIC'] = -2*df['Score'] + p*np.log(T)

    running_list_label = ["bidask_spd_USDEUR", "bidask_spd_USDGBP", "bidask_spd_USDJPY",
                      "arb_profit_USDGBP_ask", "arb_profit_USDGBP_bid", "arb_profit_USDJPY_ask",
                      "arb_profit_USDJPY_bid", "arb_profit_USDEUR_ask", "arb_profit_USDEUR_bid"]
    # running_list_label = ["bidask_spd_USDEUR"]



    df_opt_model = [[] for li in range(len(running_list_label))]
    for x,j in enumerate(running_list_label):
        df_opt_model[x] = []
        avg_BIC = []
        for i in range(2,11):
            try:
                store = pd.HDFStore("../PData/FX_transit_matrix_/"+ j + "_" + "%s"%i + ".h5")
                df = pd.read_hdf(store, j + "_"+'%s'%i)
                store.close()
                BIC(df, i)
                avg_BIC.append(np.mean(df['BIC']))
            except:
                pass
        opt_state = (avg_BIC.index(min(avg_BIC)) + 2)
        opt_model_name = "../PData/FX_transit_matrix_/"+ j + "_" + str(opt_state) + ".h5"
        df_opt_model[x] = opt_model_name
    return df_opt_model

