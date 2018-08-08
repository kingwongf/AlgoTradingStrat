import deepdish as dd
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn
from sklearn.metrics import confusion_matrix


from plotly import tools
tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.plotly as py


import pandas as pd
import numpy as np

## read price

# fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")
# xbt_data = pd.read_excel("../PData/XBTUSDEUR.xlsx", header=0, index_col ="Dates")
# cx_data_0208 = pd.read_excel("../PData/Crypto_Full_PData.xlsx", header=0, index_col ="Dates")
#
#
# xbtfx_data = fx_data.join(xbt_data)
# cxfx_data = fx_data.join(cx_data_0208)
#
# xbtfx_data = xbtfx_data.dropna()
# cxfx_data = cxfx_data.dropna()

fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")

rf = 0.0146/250


## PnL analysis

##cxfx
# cx_net_post = dd.io.load('../PData/CX_Net_pos_global.h5')
# cx_net_post = pd.DataFrame.from_dict(cx_net_post)
#
# XBTUSD_benchmark = 1000000/xbtfx_data['XBTUSD_Close_Ask'][15000+10]*xbtfx_data['XBTUSD_Close_Ask'][15000+10:15000+6048]
# XETUSD_benchmark = 1000000/cxfx_data['XETUSD_Close_Ask'][15000+10]*cxfx_data['XETUSD_Close_Ask'][15000+10:15000+6048]

# plt.plot(cx_net_post, alpha=0.7)
# plt.plot(np.arange(len(XBTUSD_benchmark)), XBTUSD_benchmark, color='red')
# plt.plot(np.arange(len(XETUSD_benchmark)), XETUSD_benchmark, color='green')
# plt.ylabel('Net Position in USD')
# plt.show()


##fx

fx_net_post = dd.io.load('../PData/FX_Net_pos_global.h5')
fx_net_post = pd.DataFrame.from_dict(fx_net_post)


USDEUR_benchmark = 1000000/fx_data['USDEUR_Close_Ask'][15000+10]*fx_data['USDEUR_Close_Ask'][15000+10:15000+6048]
USDGBP_benchmark = 1000000/fx_data['USDGBP_Close_Ask'][15000+10]*fx_data['USDGBP_Close_Ask'][15000+10:15000+6048]
USDJPY_benchmark = 1000000/fx_data['USDJPY_Close_Ask'][15000+10]*fx_data['USDJPY_Close_Ask'][15000+10:15000+6048]


plt.plot(fx_net_post, alpha=0.5)
plt.plot(np.arange(len(USDEUR_benchmark)), USDEUR_benchmark, color='red',linestyle='-.')
plt.plot(np.arange(len(USDGBP_benchmark)), USDGBP_benchmark, color='green',linestyle='-.')
plt.plot(np.arange(len(USDJPY_benchmark)), USDJPY_benchmark, color='blue',linestyle='-.')
plt.ylabel('Net Position in USD')
plt.show()



## shift analysis
cx_shift = dd.io.load('../PData/CX_Shift_global.h5')
fx_shift = dd.io.load('../PData/FX_Shift_global.h5')



def shift_byprice_backtest(price_series):
    shift =[]
    z_score = 1.645
    for i in range(10, len(price_series)):
        if price_series[i] > np.mean(price_series[i-10:i]) + np.std(price_series[i-10:i])*z_score:
            shift.append(1)
        elif price_series[i] < np.mean(price_series[i-10:i]) - np.std(price_series[i-10:i])*z_score:
            shift.append(-1)
        else:
            shift.append(0)
    return shift


def ret(df):
    ret_df = df / df.shift(1) - 1
    return ret_df


def OLS(x,y):
    linreg = sklearn.linear_model.LinearRegression(fit_intercept=True)
    linreg.fit(x,y)
    return linreg.coef_[0] , linreg.intercept_[0]


def ret_analysis(net_post_df, cxfx, P_shift):
    p_ret_df = ret(net_post_df).dropna()
    alpha_beta = pd.DataFrame([]*p_ret_df.shape[1], columns=p_ret_df.columns)
    for label in p_ret_df.columns.values:
        print(label)
        if cxfx==True:
            ticker = label[label.index('X'):label.index('X') + 6]
            if label.find("XBT") > 0:
                ask = xbtfx_data[ticker + '_Close_Ask']
                # bid = xbtfx_data[ticker + '_Close_Bid'][15000+11:15000 + 6048]

            if label.find("XET") > 0:
                ask = cxfx_data[ticker + '_Close_Ask']
                # bid = cxfx_data[ticker + '_Close_Bid'][15000+11:15000 + 6048]
        elif cxfx==False:
            try:
                ticker = label[label.index('USD'):label.index('USD') + 6]
                ask = fx_data[ticker + '_Close_Ask']

            except:
                ticker = label[label.index('EUR'):label.index('EUR') + 6]
                ask = fx_data[ticker + '_Close_Ask']
                # bid = fx_data[ticker + '_Close_Bid'][15000 + 10:15000 + 6048]


        # print(len(np.diff(ask)), len(p_ret_df[label]))

        ## alpha beta
        x = np.diff(ask[15000 + 10:15000 + 6048]) - rf/(24*12)
        y = p_ret_df[label].values
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)


        beta, alpha = OLS(x,y)


        ## Sharpe
        Sharpe = (np.mean(p_ret_df[label]) - rf/(24*12))/ np.std(p_ret_df[label])

        ## information ratio

        # print(len(p_ret_df[label] - np.diff(ask)))

        # print(np.std(p_ret_df[label].values - np.diff(ask)))

        IR = np.mean(p_ret_df[label].values - np.diff(ask[15000 + 10:15000 + 6048]))/ \
             np.std(p_ret_df[label].values - np.diff(ask[15000 + 10:15000 + 6048]))


        ## confusion matrix
        print(np.where(np.array(shift_byprice_backtest(ask[15000:15000 + 6048])) == -1)[0])

        print(len(np.where(np.array(shift_byprice_backtest(ask[15000:15000 + 6048])) == -1)[0]))
        print(len(np.where(np.array(shift_byprice_backtest(ask[15000:15000 + 6048])) == 0)[0]))
        print(len(np.where(np.array(shift_byprice_backtest(ask[15000:15000 + 6048])) == 1)[0]))

        conf_m = confusion_matrix(shift_byprice_backtest(ask[15000:15000 + 6048]), P_shift[label])
        # print(conf_m)

        alpha_beta[label] = [beta, alpha, Sharpe, IR, conf_m]

    alpha_beta_T = alpha_beta.transpose()
    alpha_beta_T.to_csv("../PData/cx_alpha_beta_T.csv")
    # alpha_beta_T = alpha_beta_T.sort_values(by=[Sharpe], ascending=False)

    # h5, not threadsafe
    file_name = "../PData/Crypto_alpha_beta_T_.h5"
    store = pd.HDFStore(file_name)
    key = "alpha_beta_T"

    alpha_beta_T.to_hdf(file_name, key=key)
    store.close()

    return alpha_beta_T



# print(list(cx_net_post.keys()))

# cx_alpha_beta = ret_analysis(cx_net_post, True,cx_shift)
fx_alpha_beta = ret_analysis(fx_net_post, False,fx_shift)

# print(cx_alpha_beta)
#
# T_cx = cx_net_post.transpose()
# T_cx.sort_values([6037],  ascending=False)


# print(cx_alpha_beta.sort_values(by=[1,2,3],ascending=[False, False, False]).head())

print(fx_alpha_beta.sort_values(by=[1,2,3],ascending=[False, False, False]).head())