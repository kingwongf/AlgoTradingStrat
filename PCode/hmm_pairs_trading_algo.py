# Ongoing work of King Wong and Ryan Mclaughlin 

import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

fx_data = pd.read_csv("hmm_test_data_set.csv", header=0)
dates = np.array(fx_data['Date_AAPL'])

ratio = np.array(fx_data['Close_AAPL'] / fx_data['Close_GOOG'])
ratio = ratio[np.logical_not(np.isnan(ratio))]

# reshape_ratio = ratio.reshape(1, -1)
full_reshape_ratio = np.array([ratio]).T
dates = np.arange(0, len(full_reshape_ratio))


mu0_list =[]
mu1_list =[]
sd0_list =[]
sd1_list =[]

P00_list = []
P01_list = []
P10_list = []
P11_list = []


orderbook = []

def hmm_fit(reshape_ratio):
    print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(reshape_ratio)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(reshape_ratio)

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


def execution(curren_mu, current_sd, timestamp):
    A = fx_data['Close_AAPL'][timestamp]
    B = fx_data['Close_GOOG'][timestamp]

    print(full_reshape_ratio[timestamp])

    if full_reshape_ratio[timestamp] < current_mu + 0.1*current_sd:
        orderbook.append('long AAPL %s, short GOOG %s' %(A,B))
    elif full_reshape_ratio[timestamp] > current_mu + 0.1* - current_sd:
        orderbook.append('short AAPL %s, long GOOG %s' % (A, B))
    else:
        orderbook.append('no trade')


for t in range(0,len(dates)-30-1):


    mu0, mu1, var0, var1, P00, P01, P10, P11 = hmm_fit(full_reshape_ratio[t:t+30])
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


    timestamp = t + 30

    current_ratio = full_reshape_ratio[timestamp]



    if abs(current_ratio - mu0) > abs(current_ratio - mu1):
        current_state = 1
    else:
        current_state = 0

    if current_state == 0:
        if P01_list[t] - P01_list[t-1] > P00_list[t] - P00_list[t-1]:     # can change condition to > P00_list[t] - P00_list[t-1], anticipating a regime shift from 0 to 1, mu and sd changes to mu1 and sqrt(var1)
            current_sd = np.sqrt(var1)
            current_mu = mu1
        else:
            current_sd = np.sqrt(var0)
            current_mu = mu0

    elif P10_list[t] - P10_list[t-1] > P11_list[t] - P11_list[t-1]:
        current_sd = np.sqrt(var0)
        current_mu = mu0

    else:
        current_sd = np.sqrt(var1)
        current_mu = mu1



    execution(current_mu, current_sd, timestamp)


for item in orderbook:
    print(item)

plot_time = range(len(np.diff(P00_list)))


plt.plot(dates, full_reshape_ratio )
plt.show()
plt.plot(plot_time, np.diff(P00_list), plot_time, np.diff(P01_list), plot_time, np.diff(P10_list), plot_time, np.diff(P11_list))
plt.show()


# fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
# colours = cm.rainbow(np.linspace(0, 1, model.n_components))
# for i, (ax, colour) in enumerate(zip(axs, colours)):
#     # Use fancy indexing to plot data in each state.
#     mask = hidden_states == i
#     ax.plot_date(dates[mask], reshape_ratio[mask], ".-", c=colour)
#     ax.set_title("{0}th hidden state".format(i))
#
#     # Format the ticks.
#     # ax.xaxis.set_major_locator(YearLocator())
#     # ax.xaxis.set_minor_locator(MonthLocator())
#
#     ax.grid(True)
# print(fig)