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


fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")


USDJPY_ratio = np.divide(np.multiply(fx_data['USDEUR_Close_Bid']
                                     ,fx_data['EURJPY_Close_Bid']), fx_data['USDJPY_Close_Bid'])

fx_data['USDJPY_ratio'] = USDJPY_ratio

USDJPY_ratio = np.array([USDJPY_ratio]).T


# mu0_list =[]
# mu1_list =[]
# sd0_list =[]
# sd1_list =[]
#
# P00_list = []
# P01_list = []
# P10_list = []
# P11_list = []




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



for n in range(2,11):
    model_score =[]
    P_list = []
    # for i in range(0, fx_data.index.get_loc('11/6/2018 19:35') -
    #                   fx_data.index.get_loc('1/1/2018 0:00') + 1): ##change it back later
    for i in range(0, fx_data.index.get_loc('1/1/2018 1:00') - fx_data.index.get_loc('1/1/2018 0:00') + 1):

        roll_window = USDJPY_ratio[
                      fx_data.index.get_loc('1/1/2018 0:00') - 15000 + i:fx_data.index.get_loc('1/1/2018 0:00') + i]


        n_state = n
        P, model_score_logprob = hmm_fit(roll_window, n_state)

        P_list.append(P)
        model_score.append(np.mean(model_score_logprob))


    P_label_list = ['P' + str(i) + str(j) for i in range(0,n_state) for j in range(0, n_state)]

    transit_matrix = pd.DataFrame(P_list, columns=P_label_list)


    transit_matrix['Score'] = model_score

    file_name = "../PData/transit_matrix_bid_usdjpy_" + n +"_states.csv"
    transit_matrix.to_csv(file_name)