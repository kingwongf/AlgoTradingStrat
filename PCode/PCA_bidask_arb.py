from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def pca(arb, bidask):
    pca = PCA(n_components=1)
    X = pd.DataFrame(data={'arb': arb, 'bidask': bidask})
    X = StandardScaler().fit_transform(X)
    principalComponent = pca.fit_transform(X)
    return principalComponent

def hmm_fit(seq_feature, n_state):
    # print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components= n_state, covariance_type="diag", n_iter=100).fit(seq_feature)

    # print(model.score(model.sample(100)[0]))

    model_score_logprob = model.score(seq_feature)

    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(seq_feature)

    # print(hidden_states)

    # print("done")
    ###############################################################################
    # Print trained parameters and plot
    # print("Transition matrix")
    # print(model.transmat_)
    # print()
    # print("Means and vars of each hidden state")
    mu =[]
    var =[]

    P = model.transmat_.flatten()
    # print(P)

    return P, model_score_logprob

def backtesting(n,vec, TICKER,  rolling):   ####TODO TICKER
    seq_to_fit = vec
    # seq_to_fit = vec.values.T
    # seq_to_fit = vec.values.reshape(1, -1)
    model_score = []
    P_list = []


    for i in range(len(seq_to_fit) - rolling):
        roll_window = seq_to_fit[i:i + rolling]
        # print(len(roll_window))
        roll_window = np.array(roll_window).reshape(-1, 1)
        # print(len(roll_window))
        P, model_score_logprob = hmm_fit(roll_window, n)
        # print(P)
        P_list.append(P)
        model_score.append(np.mean(model_score_logprob))

    P_label_list = ['P' + str(i) + str(j) for i in range(0, n) for j in range(0, n)]

    transit_matrix = pd.DataFrame(P_list, columns=P_label_list)

    transit_matrix['Score'] = model_score

    # h5
    file_name = "../PData/PCA_" + TICKER + "_" + str(n) + ".h5"
    store = pd.HDFStore(file_name)
    key = TICKER + "_" + str(n)

    transit_matrix.to_hdf(file_name, key=key)
    store.close()

    # # csv
    # file_name = str_name + "_" + str(n) + ".csv"
    #
    # transit_matrix.to_csv(file_name)

    gc.collect()

def find_opt_PCA():
    def BIC(df, n_states):
        T = 15000
        p = n_states**2 + 2*n_states -1
        df['BIC'] = -2*df['Score'] + p*np.log(T)

    avg_BIC = []
    for i in range(2,11):
        try:
            store = pd.HDFStore("../PData/PCA_"+ "TICKER" + "_" + "%s"%i + ".h5")                    ####TODO add TICKER
            df = pd.read_hdf(store, "TICKER" + "_"+'%s'%i)
            store.close()
        except:
            pass
        BIC(df, i)
        avg_BIC.append(np.mean(df['BIC']))
    opt_state = (avg_BIC.index(min(avg_BIC)) + 2)
    opt_model_name = "../PData/PCA_"+ "TICKER" + "_" + str(opt_state) + ".h5"
    return opt_model_name


def mainPCA():
    states_list = range(2,11)
    with Pool() as pool:
        pool.starmap(backtesting,
                     zip(states_list, repeat(pca(arb= , bidask= )), repeat("TICKER"), repeat(15000)))   ####TODO add TICKER

    opt_PCA_model_name = find_opt_PCA()

    store = pd.HDFStore(opt_PCA_model_name)
    df = pd.read_hdf(store, opt_PCA_model_name[13:-3])  ## drop name of the directory to ".h5"
    store.close()
    transition_matrix = df.drop(labels='Score', axis=1)
    list_transit = transition_matrix.T.values.tolist()
    print(list_transit)
    PCA_alphas = (list_transit)  ## list in the order of find_opt_model


    from algo_trade_backtest import main

    PnL_PCA = {}

    for P in PCA_alphas:
        PnL = main(P=P, bid=bid, ask=ask)
        PnL_PCA[label] = PnL

if __name__=="__main__":
    freeze_support()
    mainPCA()



