import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
import pandas as pd
import gc
import warnings
import gc
warnings.filterwarnings("ignore", category=DeprecationWarning)


def hmm_fit(seq_feature, n_state):
    print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components= n_state, covariance_type="diag", n_iter=1000).fit(seq_feature)

    # print(model.score(model.sample(100)[0]))

    model_score_logprob = model.score(seq_feature)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(seq_feature)

    # print(hidden_states)

    print("done")
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
arb_profit_USDEUR_ask = np.multiply(1/xbtfx_data['XBTUSD_Close_Bid'], xbtfx_data['XBTEUR_Close_Ask']) - \
                        xbtfx_data['USDEUR_Close_Ask']

arb_profit_USDEUR_bid = np.multiply(1/xbtfx_data['XBTUSD_Close_Ask'], xbtfx_data['XBTEUR_Close_Bid']) - \
                        xbtfx_data['USDEUR_Close_Bid']


running_list = [bidask_spd_XBTUSD, bidask_spd_XBTEUR, bidask_spd_XETUSD, bidask_spd_XRPUSD,
                bidask_spd_XETEUR, arb_profit_USDEUR_ask, arb_profit_USDEUR_bid]


running_list_label = ["bidask_spd_XBTUSD", "bidask_spd_XBTEUR", "bidask_spd_XETUSD", "bidask_spd_XRPUSD",
                      "bidask_spd_XETEUR", "arb_profit_USDEUR_ask", "arb_profit_USDEUR_bid"]

dfrun = pd.DataFrame([running_list], columns=running_list_label)

rolling=100

gc.enable()
for vec in dfrun:
    # print(dfrun[vec])
    seq_to_fit = np.array(dfrun[vec].tolist()).T
    print(seq_to_fit)

    for n in range(2,11):
        model_score =[]
        P_list = []

        for i in range(len(seq_to_fit - rolling)):

            roll_window = seq_to_fit[i:i+rolling]

            P, model_score_logprob = hmm_fit(roll_window, n)

            print(P)

            P_list.append(P)
            model_score.append(np.mean(model_score_logprob))


        P_label_list = ['P' + str(i) + str(j) for i in range(0,n) for j in range(0,n)]

        transit_matrix = pd.DataFrame(P_list, columns=P_label_list)

        transit_matrix['Score'] = model_score
        print(vec)

        file_name = "../PData/transit_matrix/Cypto_transit_matrix_" + str(vec) + "_" + str(n) + "_states.h5"
        transit_matrix.to_hdf(file_name)

        print(str(12 - n) + " out of 11 states model left to fit and " +
              str(len(dfrun.columns) - dfrun.columns.get_loc(vec)) + " sequences left")
