import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.base import _BaseHMM
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
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


running_list = [bidask_spd_USDEUR, bidask_spd_USDGBP, bidask_spd_USDJPY, bidask_spd_EURJPY, bidask_spd_EURGBP,
                arb_profit_USDGBP_ask, arb_profit_USDGBP_bid, arb_profit_USDJPY_ask, arb_profit_USDJPY_bid]

running_list_label = ["bidask_spd_USDEUR", "bidask_spd_USDGBP", "bidask_spd_USDJPY", "bidask_spd_EURJPY",
                      "bidask_spd_EURGBP", "arb_profit_USDGBP_ask", "arb_profit_USDGBP_bid",
                      "arb_profit_USDJPY_ask", "arb_profit_USDJPY_bid"]

dfrun = pd.DataFrame([running_list], columns=running_list_label)



for vec in dfrun:
    seq_to_fit = np.array(dfrun[vec].tolist()).T
    for n in range(2,11):
        model_score =[]
        P_list = []
        # for i in range(0, fx_data.index.get_loc('11/6/2018 19:35') -
        #                   fx_data.index.get_loc('1/1/2018 0:00') + 1): ##TODO change it back later
        for i in range(0, fx_data.index.get_loc('1/1/2018 1:00') - fx_data.index.get_loc('1/1/2018 0:00') + 1):

            roll_window = seq_to_fit[i:fx_data.index.get_loc('1/1/2018 0:00') + i]

            n_state = n # number of states fitted

            P, model_score_logprob = hmm_fit(roll_window, n_state)

            P_list.append(P)
            model_score.append(np.mean(model_score_logprob))


        P_label_list = ['P' + str(i) + str(j) for i in range(0,n_state) for j in range(0, n_state)]

        transit_matrix = pd.DataFrame(P_list, columns=P_label_list)


        transit_matrix['Score'] = model_score
        print(vec)

        file_name = "../PData/transit_matrix/transit_matrix_" + str(vec) + "_" + str(n) + "_states.csv"
        transit_matrix.to_csv(file_name)

        print(str(12 - n) + " out of 11 states model left to fit and " +
              str(len(dfrun.columns) - dfrun.columns.get_loc(vec)) + " sequences left")