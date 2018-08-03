import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plotly import tools
tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.plotly as py

import plotly.graph_objs as go

z_score = 1.645



def shift_byret(price_series):
    ret = np.diff(np.log(price_series))
    shift = [0]*(len(ret)-10)

    for ind in range(10, len(ret)-10):
        if ret[ind] > np.mean(ret) + np.std(ret)*z_score:
            shift[ind] = 1
        if ret[ind] < np.mean(ret) - np.std(ret)*z_score:
            shift[ind] = -1
    return shift


def shift_byret_backtest(price_series):
    shift =[]
    ret = np.diff(np.log(price_series))
    for i in range(10, len(ret)):
        # print(i)
        if ret[i] > np.mean(ret[i-10:i]) + np.std(ret[i-10:i])*z_score:
            shift.append(1)
        elif ret[i] < np.mean(ret[i-10:i]) - np.std(ret[i-10:i])*z_score:
            shift.append(-1)
        else:
            shift.append(0)
    # shift = [0] + shift
    return shift

def shift_byprice_backtest(price_series):
    shift =[]
    for i in range(10, len(price_series)):
        if price_series[i] > np.mean(price_series[i-10:i]) + np.std(price_series[i-10:i])*z_score:
            shift.append(1)
        elif price_series[i] < np.mean(price_series[i-10:i]) - np.std(price_series[i-10:i])*z_score:
            shift.append(-1)
        else:
            shift.append(0)
    return shift[1::]





r = 0.01
q = 0
sig = 0.1
lamb = 1
mu_j = 0.01
sig_j = 0.3

g = r - q - 0.5*sig**2 - lamb*(np.exp(mu_j + 0.5*(sig_j)**2)-1)
test_price = [100] + [0]*500

dt = 1/250
for i in range(1,len(test_price)):

    P = np.random.poisson(lam=lamb*dt)
    U = np.exp(P * mu_j + np.sqrt(P)*sig_j*np.random.normal())
    test_price[i] = test_price[i-1]*np.exp((r -lamb*(np.exp(mu_j+0.5*(sig_j**2)) - 1) - 0.5 * sig**2)*dt + sig*np.sqrt(dt)*np.random.normal())*U
    # test_price[i] = test_price[i-1]*np.exp(g + sig*np.random.normal() + np.sum([np.random.normal(mu_j, sig_j) for i in range()]))

print(test_price[-1])

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
x = np.arange(len(test_price[11::]))
ax1.plot(x, shift_byret(test_price), label= 'by return', alpha=0.7)
ax1.plot(x, shift_byret_backtest(test_price), label = 'by backtest return', alpha=0.7)
ax1.plot(x, shift_byprice_backtest(test_price), label = 'by backtest price', alpha=0.7)
ax2.plot(x, test_price[11::], 'r-')

ax1.set_xlabel('date')
ax1.set_ylabel('shift')
ax2.set_ylabel('Price')
ax1.legend()
plt.show()

## plotly

## graphing/ plotting

shift_byret_plot = go.Scatter(x=x,    y=shift_byret(test_price))

shift_byret_backtest_plot = go.Scatter(x=x, y=shift_byret_backtest(test_price), name='Shift detection by return')

shift_byprice_backtest_plot = go.Scatter(x=x,    y=shift_byprice_backtest(test_price), name='Shift detection by price')

test_price_plot = go.Scatter(
    x=x,
    y=test_price[11::], name="Price")

data = [test_price_plot, shift_byret_plot, shift_byret_backtest_plot, shift_byprice_backtest_plot]

fig = tools.make_subplots(rows=4, cols=1,shared_xaxes=True)

fig.append_trace(test_price_plot, 1,1)
# fig.append_trace(shift_byret_plot, 2,1)
fig.append_trace(shift_byret_backtest_plot, 2,1)
fig.append_trace(shift_byprice_backtest_plot, 3,1)
# for ind in range(1,5):
#     fig.append_trace(data[ind-1], ind, 1)


fig['layout'].update(title='Jump Diffusion Price Simmulation with Structural Break Detection')
py.iplot(fig, filename='Jump Diffusion Price Simmulation with Structural Break Detection')