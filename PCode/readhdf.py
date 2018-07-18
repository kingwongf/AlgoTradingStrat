import pandas as pd

store = pd.HDFStore("../PData/Crypto_transit_matrix_/bidask_spd_XBTUSD_3.h5")
df = pd.read_hdf(store, 'bidask_spd_XBTUSD_3')


store.close()


print(df)