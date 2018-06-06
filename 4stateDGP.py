# Ongoing work of King Wong and Ryan Mclaughlin

## this is to test/ justify using the first difference of the prior probabilities
# to forecast an imminent structural break in the time series

import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


## simple DGP of Gaussian process

dgp_mu0 = 0.1
dgp_mu1 = 0.3
dgp_mu2 = 0.2
dgp_mu3 = 0.4


dgp_std0 = 0.05
dgp_std1 = 0.04
dgp_std2 = 0.05
dgp_std3 = 0.04



t = np.random.random_integers(0,1000,7)

z0 = dgp_mu0 + dgp_std0*np.random.rand(t[0])
z1 = dgp_mu1 + dgp_std1*np.random.rand(t[1])
z2 = dgp_mu2 + dgp_std2*np.random.rand(t[2])
z3 = dgp_mu1 + dgp_std1*np.random.rand(t[3])
z4 = dgp_mu3 + dgp_std3*np.random.rand(t[4])
z5 = dgp_mu2 + dgp_std2*np.random.rand(t[5])
z6 = dgp_mu0 + dgp_std0*np.random.rand(t[6])

z = np.concatenate((z0,z1,z2,z3,z4,z5,z6))

# reshape_ratio = ratio.reshape(1, -1)
z = np.array([z]).T
dates = np.arange(0, len(z))

mu0_list =[]
mu1_list =[]
sd0_list =[]
sd1_list =[]

P_list = []




def hmm_fit(reshape_ratio):
    print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(reshape_ratio)

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

    P = model.transmat_

    for i in range(model.n_components):
        #print("{0}th hidden state".format(i))
        #print("mean = ", i, model.means_[i])
        mu.append(model.means_[i])
        #print("var = ", np.diag(model.covars_[i]))
        var.append(np.diag(model.covars_[i]))
        #print()
    return mu[0], mu[1], var[0], var[1], P


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


for t in range(0,len(dates)-10-1):


    mu0, mu1, var0, var1, P = hmm_fit(z[t:t+100])
    hold = [mu0, mu1, var0, var1]

    # print(mu0, mu1)

    mu0_list.append(mu0)
    mu1_list.append(mu1)
    sd0_list.append(np.sqrt(var0))
    sd1_list.append(np.sqrt(var1))
    P_list.append(P)


    timestamp = t + 30




    # execution(current_mu, current_sd, timestamp)


plot_time = range(len(np.diff(P_list,axis=0)))

print(np.diff(P_list, axis=0).shape)

x3D, y3D, z3D = np.diff(P_list, axis=0).nonzero()


plt.plot(dates, z)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x3D,y3D,z3D, cmap = cm.jet)
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