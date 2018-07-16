import pandas as pd

store = pd.HDFStore("../PData/Crypto_transit_matrix_/bidask_spd_XBTUSD_10_states.h5")
df = pd.read_hdf(store, 'bidask_spd_XBTUSD_10')


store.close()


print(df)