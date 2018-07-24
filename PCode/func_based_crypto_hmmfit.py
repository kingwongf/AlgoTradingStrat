import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
import pandas as pd
import itertools
import warnings
import gc
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

def backtesting(n,vec, str_name,  rolling):
    seq_to_fit = vec
    # seq_to_fit = vec.values.T
    # seq_to_fit = vec.values.reshape(1, -1)
    model_score = []
    P_list = []

    for i in range(6048):      ## backtesting the first 6,048 models, approx 1 month = 21 days * 24hr/day * 60min/ hr / 5min interval
    # for i in range(len(seq_to_fit) - rolling):    ## full backtesting window
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

    # h5, not threadsafe
    file_name = "../PData/Crypto_transit_matrix_/6048_backtesting/" + str_name + "_" + str(n) + ".h5"
    store = pd.HDFStore(file_name)
    key = str_name + "_" + str(n)

    transit_matrix.to_hdf(file_name, key=key)
    store.close()

    # # csv
    # file_name = str_name + "_" + str(n) + ".csv"
    #
    # transit_matrix.to_csv(file_name)

    # gc.collect()



fx_data = pd.read_excel("../PData/FX_PData.xlsx", header=0, index_col ="Dates")
xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")


xbtfx_data = fx_data.join(xbt_data)
cxfx_data = fx_data.join(cx_data_0208)

xbtfx_data = xbtfx_data.dropna()
cxfx_data = cxfx_data.dropna()



# print(cxfx_data.index.values)
#                     # "2017-11-09T22:00:00.000000000
# print(cxfx_data.index.get_loc('2017-02-08T04:40:00.000000000'))
#
# cxfx_data = cxfx_data.loc["2017-02-08T04:40:00.000000000":]


## data start on 09/11/2017  22:00:00
bidask_spd_XBTUSD = xbtfx_data['XBTUSD_Close_Ask'] - xbtfx_data['XBTUSD_Close_Bid']

bidask_spd_XBTEUR = xbtfx_data['XBTEUR_Close_Ask'] - xbtfx_data['XBTEUR_Close_Bid']



## data start on 08/02/2018  04:40:00

bidask_spd_XETUSD = cxfx_data['XETUSD_Close_Ask'] - cxfx_data['XETUSD_Close_Bid']

bidask_spd_XRPUSD = cxfx_data['XRPUSD_Close_Ask'] - cxfx_data['XRPUSD_Close_Bid']

bidask_spd_XETEUR = cxfx_data['XETEUR_Close_Ask'] - cxfx_data['XETEUR_Close_Bid']


## data start on 09/11/2017  22:00:00
arb_profit_USDEUR_ask_fromXBT = np.multiply(1/xbtfx_data['XBTUSD_Close_Bid'], xbtfx_data['XBTEUR_Close_Ask']) - \
                        xbtfx_data['USDEUR_Close_Ask']

arb_profit_USDEUR_bid_fromXBT = np.multiply(1/xbtfx_data['XBTUSD_Close_Ask'], xbtfx_data['XBTEUR_Close_Bid']) - \
                        xbtfx_data['USDEUR_Close_Bid']


arb_profit_USDEUR_ask_fromXET = np.multiply(1/cxfx_data['XETUSD_Close_Bid'], cxfx_data['XETEUR_Close_Ask']) - \
                        cxfx_data['USDEUR_Close_Ask']

arb_profit_USDEUR_bid_fromXET = np.multiply(1/cxfx_data['XETUSD_Close_Ask'], cxfx_data['XETEUR_Close_Bid']) - \
                        cxfx_data['USDEUR_Close_Bid']


# running_list = [bidask_spd_XBTUSD, bidask_spd_XBTEUR, bidask_spd_XETUSD, bidask_spd_XRPUSD,
#                 bidask_spd_XETEUR, arb_profit_USDEUR_ask, arb_profit_USDEUR_bid]

running_list = [bidask_spd_XETUSD, bidask_spd_XRPUSD,
                bidask_spd_XETEUR, arb_profit_USDEUR_ask_fromXBT,
                arb_profit_USDEUR_bid_fromXBT, arb_profit_USDEUR_ask_fromXET]


# running_list_label = ["bidask_spd_XBTUSD", "bidask_spd_XBTEUR", "bidask_spd_XETUSD", "bidask_spd_XRPUSD",
#                       "bidask_spd_XETEUR", "arb_profit_USDEUR_ask", "arb_profit_USDEUR_bid"]

running_list_label = ["bidask_spd_XETUSD", "bidask_spd_XRPUSD",
                      "bidask_spd_XETEUR", "arb_profit_USDEUR_ask_fromXBT",
                      "arb_profit_USDEUR_bid_fromXBT","arb_profit_USDEUR_ask_fromXET"]

# dfrun = pd.DataFrame([running_list], columns=running_list_label)



gc.enable()



def main():
    states_list = range(2, 11)
    # backtesting(9, bidask_spd_XBTEUR, "bidask_spd_XBTEUR", 15000)
    # backtesting(10, bidask_spd_XBTEUR, "bidask_spd_XBTEUR", 15000)
    # for li in range(len(running_list)):
    #     for lj in states_list:
    #         backtesting(lj, running_list[li], running_list_label[li], 15000)

    # run parellel
    with Pool() as pool:
        pool.starmap(backtesting,
                       zip(states_list, repeat(bidask_spd_XRPUSD), repeat("bidask_spd_XRPUSD"), repeat(15000)))
        # for seq in range(len(running_list)):
        #      pool.starmap(backtesting, zip(states_list, repeat(running_list[seq]),repeat(running_list_label[seq]), repeat(15000)))

if __name__=="__main__":
    freeze_support()
    main()




# backtesting(2, bidask_spd_XBTUSD, "bidask_spd_XBTUSD", 44894)

# print(hmm_fit(bidask_spd_XBTUSD, 2))