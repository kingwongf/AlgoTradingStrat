import numpy as np
np.set_printoptions(threshold=np.nan)

import plotly as py
py.tools.set_credentials_file(username='kingwongf', api_key='vwqbsMCcdGLvf5LNkCRK')
import plotly.graph_objs as go
import pandas as pd



fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")

