import pandas as pd

store = pd.HDFStore("../PData/Crypto_transit_matrix_/6048_backtesting/bidask_spd_XETUSD_10.h5")
df = pd.read_hdf(store, 'bidask_spd_XETUSD_10')


store.close()


print(df)