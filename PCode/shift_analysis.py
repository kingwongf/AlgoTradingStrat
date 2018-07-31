import numpy as np
import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go



def shift_byret(price_series):
    ret = np.diff(np.log(price_series))
    shift = [0]*(len(ret)-10)
    print(len(shift))
    for ind in range(10, len(ret)-10):
        if ret[ind] > np.mean(ret) + np.std(ret):
            shift[ind] = 1
        if ret[ind] < np.mean(ret) - np.std(ret):
            shift[ind] = -1
    return len(shift)


def shift_byret_backtest(price_series):
    shift =[]
    ret = np.diff(np.log(price_series))
    for i in range(10, len(price_series)):
        if ret[i] > np.mean(ret[i-10:i]) + np.std(ret[i-10:i]):
            shift.append(1)
        if ret[i] < np.mean(ret[i-10:i]) - np.std(ret[i-10:i]):
            shift.append(-1)
        else:
            shift.append(0)
    # shift = [0]*10+shift
    return len(shift)

def shift_byprice_backtest(price_series):
    shift =[]
    for i in range(10, len(price_series)):
        if price_series[i] > np.mean(price_series[i-10:i]) + np.std(price_series[i-10:i]):
            shift.append(1)
        if price_series[i] < np.mean(price_series[i-10:i]) - np.std(price_series[i-10:i]):
            shift.append(-1)
        else:
            shift.append(0)
    shift = [0] + shift
    return len(shift)


test_price = [100,97,88,98,80,45,30,25,24,23,90,105,80,45,30,25,24,23,90,105,88,98,80,45,30,25,24,23,90]



print(shift_byret(test_price))
print(shift_byprice_backtest(test_price))
print(shift_byret_backtest(test_price))

print(len(test_price[10::]))