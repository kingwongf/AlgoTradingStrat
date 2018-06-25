import pandas as pd

fx_data = pd.read_csv("../PData/FX_PData.csv", header=0, index_col ="Dates")
macronews = pd.read_csv("../PData/MacroNews09_11_2017_11_06_2018.csv", header=0, index_col ="DateTime")

fx_news_concatenate = pd.concat([macronews,fx_data],axis=1)

print(fx_news_concatenate.head())