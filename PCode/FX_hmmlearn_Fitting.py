import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


fx_data = pd.read_csv("FX_PData.csv", header=0, index_col ="Dates")


USDGBP_ratio = np.divide(np.multiply(fx_data['USDEUR_Close_Ask'],fx_data['EURGBP_Close_Ask']), fx_data['USDGBP_Close_Ask'])

fx_data['USDGBP_ratio'] = USDGBP_ratio

USDGBP_ratio = np.array([USDGBP_ratio]).T


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


for i in range(0, fx_data.index.get_loc('9/3/2018 19:35') - fx_data.index.get_loc('9/1/2018') + 1):

    roll_window = USDGBP_ratio[fx_data.index.get_loc('9/1/2018') - 10000 + i:fx_data.index.get_loc('9/1/2018') + i]

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

    # current_ratio = usdeur_ratio[timestamp]
    #
    #
    #
    # if abs(current_ratio - mu0) > abs(current_ratio - mu1):
    #     current_state = 1
    # else:
    #     current_state = 0
    #
    # if current_state == 0:
    #     if P01_list[t] - P01_list[t-1] > P00_list[t] - P00_list[t-1]:     # can change condition to > P00_list[t] - P00_list[t-1], anticipating a regime shift from 0 to 1, mu and sd changes to mu1 and sqrt(var1)
    #         current_sd = np.sqrt(var1)
    #         current_mu = mu1
    #     else:
    #         current_sd = np.sqrt(var0)
    #         current_mu = mu0
    #
    # elif P10_list[t] - P10_list[t-1] > P11_list[t] - P11_list[t-1]:
    #     current_sd = np.sqrt(var0)
    #     current_mu = mu0
    #
    # else:
    #     current_sd = np.sqrt(var1)
    #     current_mu = mu1
    #
    # # execution(current_mu, current_sd, timestamp)




plot_time = fx_data.index[fx_data.index.get_loc('9/1/2018'):fx_data.index.get_loc('9/3/2018 19:35')]
plt.subplot(2,1,1)
plt.plot(fx_data['USDGBP_ratio'].loc['9/1/2018':'9/3/2018 19:35'])
plt.subplot(2,1,2)
plt.plot(plot_time, np.diff(P00_list), plot_time, np.diff(P01_list), plot_time, np.diff(P10_list), plot_time, np.diff(P11_list))
plt.savefig('USDGBP_ratio.png')