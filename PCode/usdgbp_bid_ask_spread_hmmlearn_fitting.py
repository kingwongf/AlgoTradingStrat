import numpy as np
from hmmlearn.hmm import GaussianHMM
import pandas as pd


fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")


USDGBP_bidask_spd = fx_data["USDGBP_Close_Ask"] - fx_data['USDGBP_Close_Bid']

fx_data['USDGBP_bidask_spd'] = USDGBP_bidask_spd

USDGBP_bidask_spd = np.array([USDGBP_bidask_spd]).T


mu0_list =[]
mu1_list =[]
sd0_list =[]
sd1_list =[]

P00_list = []
P01_list = []
P10_list = []
P11_list = []


def hmm_fit(reshape_ratio):
    print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(reshape_ratio)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(reshape_ratio)

    print(hidden_states)

    print("done")
    ###############################################################################
    # Print trained parameters and plot
    print("Transition matrix")
    print(model.transmat_)
    # print()
    # print("Means and vars of each hidden state")
    mu =[]
    var =[]

    P00 = model.transmat_[0][0]
    P01 = model.transmat_[0][1]
    P10 = model.transmat_[1][0]
    P11 = model.transmat_[1][1]

    for i in range(model.n_components):
        #print("{0}th hidden state".format(i))
        #print("mean = ", i, model.means_[i])
        mu.append(model.means_[i])
        #print("var = ", np.diag(model.covars_[i]))
        var.append(np.diag(model.covars_[i]))
        #print()
    return mu[0], mu[1], var[0], var[1], P00, P01, P10, P11


for i in range(0, fx_data.index.get_loc('11/6/2018 19:35') - fx_data.index.get_loc('1/1/2018 0:00') + 1):

    roll_window = USDGBP_bidask_spd[fx_data.index.get_loc('1/1/2018 0:00') - 15000 + i:fx_data.index.get_loc('1/1/2018 0:00') + i]

    mu0, mu1, var0, var1, P00, P01, P10, P11 = hmm_fit(roll_window)

    hold = [mu0, mu1, var0, var1, P00, P01, P10, P11]

    # print(mu0, mu1)

    mu0_list.append(mu0)
    mu1_list.append(mu1)
    sd0_list.append(np.sqrt(var0))
    sd1_list.append(np.sqrt(var1))
    P00_list.append(P00)
    P01_list.append(P01)
    P10_list.append(P10)
    P11_list.append(P11)


transit_matrix = pd.DataFrame(
    {'P00': P00_list,
     'P01': P01_list,
     'P10': P10_list,
     'P11' : P11_list
    })

transit_matrix.to_csv("../PData/transit_matrix_bid_ask_spd_usdgbp.csv")