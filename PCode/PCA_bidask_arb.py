from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
import pandas as pd
import itertools
import warnings
import gc
from functools import partial
from itertools import repeat
import deepdish as dd
from multiprocessing import Pool, freeze_support
warnings.filterwarnings("ignore", category=DeprecationWarning)



def pca(arb, bidask):
    pca = PCA(n_components=1)
    X = pd.DataFrame(data={'arb': arb, 'bidask': bidask})
    X = StandardScaler().fit_transform(X)
    principalComponent = pca.fit_transform(X)
    return principalComponent

def hmm_fit(seq_feature, n_state):
    # print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components= n_state, covariance_type="diag", n_iter=100).fit(seq_feature)

    # print(model.score(model.sample(100)[0]))

    model_score_logprob = model.score(seq_feature)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(seq_feature)

    # print(hidden_states)

    # print("done")
    ###############################################################################
    # Print trained parameters and plot
    # print("Transition matrix")
    # print(model.transmat_)
    # print()
    # print("Means and vars of each hidden state")
    mu =[]
    var =[]

    P = model.transmat_.flatten()
    # print(P)

    return P, model_score_logprob

def backtesting(n,vec, TICKER,  rolling):   ####TODO TICKER
    seq_to_fit = vec
    # seq_to_fit = vec.values.T
    # seq_to_fit = vec.values.reshape(1, -1)
    model_score = []
    P_list = []


    for i in range(len(seq_to_fit) - rolling):
        roll_window = seq_to_fit[i:i + rolling]
        # print(len(roll_window))
        roll_window = np.array(roll_window).reshape(-1, 1)
        # print(len(roll_window))
        P, model_score_logprob = hmm_fit(roll_window, n)
        # print(P)
        P_list.append(P)
        model_score.append(np.mean(model_score_logprob))

    P_label_list = ['P' + str(i) + str(j) for i in range(0, n) for j in range(0, n)]

    transit_matrix = pd.DataFrame(P_list, columns=P_label_list)

    transit_matrix['Score'] = model_score

    # h5
    file_name = "../PData/PCA_" + TICKER + "_" + str(n) + ".h5"
    store = pd.HDFStore(file_name)
    key = TICKER + "_" + str(n)

    transit_matrix.to_hdf(file_name, key=key)
    store.close()

    # # csv
    # file_name = str_name + "_" + str(n) + ".csv"
    #
    # transit_matrix.to_csv(file_name)

    gc.collect()

def find_opt_PCA(TICKER):
    def BIC(df, n_states):
        T = 15000
        p = n_states**2 + 2*n_states -1
        df['BIC'] = -2*df['Score'] + p*np.log(T)

    avg_BIC = []
    for i in range(2,11):
        try:
            store = pd.HDFStore("../PData/PCA_"+ TICKER + "_" + "%s"%i + ".h5")                    ####TODO add TICKER
            df = pd.read_hdf(store, TICKER + "_"+'%s'%i)
            store.close()
        except:
            pass
        BIC(df, i)
        avg_BIC.append(np.mean(df['BIC']))
    opt_state = (avg_BIC.index(min(avg_BIC)) + 2)
    opt_model_name = "../PData/PCA_"+ "TICKER" + "_" + str(opt_state) + ".h5"
    return opt_model_name

## loading data

fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")


xbtfx_data = fx_data.join(xbt_data)
cxfx_data = fx_data.join(cx_data_0208)

xbtfx_data = xbtfx_data.dropna()
cxfx_data = cxfx_data.dropna()

arb_profit_XBTUSD_ask = np.multiply(1/xbtfx_data['USDEUR_Close_Bid'], xbtfx_data['XBTEUR_Close_Ask']) - \
                        xbtfx_data['XBTUSD_Close_Ask']

bidask_spd_XBTUSD = xbtfx_data['XBTUSD_Close_Ask'] - xbtfx_data['XBTUSD_Close_Bid']


def mainPCA():
    states_list = range(2,11)
    with Pool() as pool:
        pool.starmap(backtesting,
                     zip(states_list, repeat(pca(arb= arb_profit_XBTUSD_ask, bidask= bidask_spd_XBTUSD)), repeat("XBTUSD"), repeat(15000)))   ####TODO add TICKER

    opt_PCA_model_name = find_opt_PCA('XBTUSD')

    store = pd.HDFStore(opt_PCA_model_name)
    df = pd.read_hdf(store, opt_PCA_model_name[13:-3])  ## drop name of the directory to ".h5"
    store.close()

    transition_matrix = df.drop(labels='Score', axis=1)
    list_transit = transition_matrix.T.values.tolist()

    ### backtesting

    from algo_trade_backtest import main

    Net_pos_global = {}
    shift_global = {}

    print(list_transit)


    ask = xbtfx_data['XBTUSD_Close_Ask'][15000 + 10:15000 + 6048] ### TODO change ticker when switching security
    bid = xbtfx_data['XBTUSD_Close_Bid'][15000 + 10:15000 + 6048]

    ## if xbt: use xbtfx, if xetusd/eur: use cxfx

    P_label_list = ['P' + str(i) + str(j) for i in range(0, len(list_transit)) for j in range(0, len(list_transit))]

    for ind, P in enumerate(list_transit):
        Net_pos, shift = main(P=P, bid=bid, ask=ask)
        Net_pos_global["PCA_XBTUSD" + "_" + P_label_list[ind]] = Net_pos
        shift_global["PCA_XBTUSD" + "_" + P_label_list[ind]] = shift

    # print(transit_global['bidask_spd_XBTUSD_10'])

    dd.io.save('../PData/PCA_XBTUSD_Net_pos_global.h5', Net_pos_global, compression=None)
    dd.io.save('../PData/PCA_XBTUSD_Shift_global.h5', shift_global, compression=None)


if __name__=="__main__":
    freeze_support()
    mainPCA()



