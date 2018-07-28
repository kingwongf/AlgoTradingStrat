import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
import pandas as pd
import gc
import itertools
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import warnings
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
    file_name = "../PData/FX_transit_matrix_/" + str_name + "_" + str(n) + ".h5"
    store = pd.HDFStore(file_name)
    key = str_name + "_" + str(n)

    transit_matrix.to_hdf(file_name, key=key)
    store.close()

    # # csv
    # file_name = str_name + "_" + str(n) + ".csv"
    #
    # transit_matrix.to_csv(file_name)

    gc.collect()



fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")

bidask_spd_USDEUR = fx_data['USDEUR_Close_Ask'] - fx_data['USDEUR_Close_Bid']

bidask_spd_USDGBP = fx_data['USDGBP_Close_Ask'] - fx_data['USDGBP_Close_Bid']

bidask_spd_USDJPY = fx_data['USDJPY_Close_Ask'] - fx_data['USDJPY_Close_Bid']

bidask_spd_EURJPY = fx_data['EURJPY_Close_Ask'] - fx_data['EURJPY_Close_Bid']

bidask_spd_EURGBP = fx_data['EURGBP_Close_Ask'] - fx_data['EURGBP_Close_Bid']


arb_profit_USDGBP_ask = np.multiply(fx_data['USDEUR_Close_Ask'], fx_data['EURGBP_Close_Ask']) - \
                        fx_data['USDGBP_Close_Ask']

arb_profit_USDGBP_bid = np.multiply(fx_data['USDEUR_Close_Bid'], fx_data['EURGBP_Close_Bid']) - \
                        fx_data['USDGBP_Close_Bid']

arb_profit_USDJPY_ask = np.multiply(fx_data['USDEUR_Close_Ask'], fx_data['EURJPY_Close_Ask']) - \
                        fx_data['USDJPY_Close_Ask']

arb_profit_USDJPY_bid = np.multiply(fx_data['USDEUR_Close_Bid'], fx_data['EURJPY_Close_Bid']) - \
                        fx_data['USDJPY_Close_Bid']

arb_profit_USDEUR_ask = np.multiply(np.divide(1,fx_data['EURJPY_Close_Bid']), fx_data['USDJPY_Close_Ask']) - \
                        fx_data['USDJPY_Close_Ask']

arb_profit_USDEUR_bid = np.multiply(np.divide(1,fx_data['EURJPY_Close_Ask']), fx_data['USDJPY_Close_Bid']) - \
                        fx_data['USDJPY_Close_Bid']


running_list = [bidask_spd_USDEUR, bidask_spd_USDGBP, bidask_spd_USDJPY,
                arb_profit_USDGBP_ask, arb_profit_USDGBP_bid, arb_profit_USDJPY_ask,
                arb_profit_USDJPY_bid, arb_profit_USDEUR_ask, arb_profit_USDEUR_bid,
                bidask_spd_EURJPY, bidask_spd_EURGBP]

running_list_label = ["bidask_spd_USDEUR", "bidask_spd_USDGBP", "bidask_spd_USDJPY",
                      "arb_profit_USDGBP_ask", "arb_profit_USDGBP_bid", "arb_profit_USDJPY_ask",
                      "arb_profit_USDJPY_bid", "arb_profit_USDEUR_ask", "arb_profit_USDEUR_bid",
                      "bidask_spd_EURJPY", "bidask_spd_EURGBP"]




gc.enable()





## run parellel
def main():
    states_list = range(4,11)
    with Pool() as pool:
        # for seq in range(len(running_list)):
        #     pool.starmap(backtesting, zip(states_list, repeat(running_list[seq]),repeat(running_list_label[seq]), repeat(15000)))
        pool.starmap(backtesting,
                     zip(states_list, repeat(bidask_spd_USDJPY), repeat("bidask_spd_USDJPY"), repeat(15000)))

if __name__=="__main__":
    freeze_support()
    main()
