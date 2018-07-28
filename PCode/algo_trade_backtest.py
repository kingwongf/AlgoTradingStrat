import numpy as np
import pandas as pd


def add(x,y):
    return x+y
## 1: higher mu, 0: lower mu

def net_position(holdings, notional, curr_price):
    net = holdings[-1]*curr_price + notional[-1]
    return net


def order(long_short, bid_ind, ask_ind, long_order, short_order,
          check, holdings, notional, number_of_currency = 0.1):
    transact = 0.0030
    if long_short == True:
        if check ==True:
            return (number_of_currency * ask_ind*(1+transact))
        else:
            holdings.append(holdings[-1] + number_of_currency)
            notional.append(notional[-1] - number_of_currency*ask_ind*(1+transact))
            long_order.append(number_of_currency*ask_ind)

    elif check==True:
        return (-number_of_currency * bid_ind*(1+transact))
    else:
        holdings.append(holdings[-1] - number_of_currency)
        notional.append(notional[-1] + number_of_currency*bid_ind)
        short_order.append(-number_of_currency*bid_ind)


def main(P, bid, ask):
    PnL = []
    long_order = []
    short_order = []
    holdings = [0]
    notional = [1000000] ## USD/ EUR
    position_limit = 1000000 + 500000
    for i in range(10, len(P)-10):
        PnL.append(net_position(holdings, notional, ask[i]))
        assert -position_limit <= net_position(holdings, notional, ask[i])
        if P[i] > np.mean(P[i-10:i]) + np.std(P[i-10:i]):
            if net_position(holdings, notional, ask[i]) + \
                    order(True, bid[i], ask[i], long_order, short_order, True, holdings, notional) < position_limit:
                order(True, bid[i], ask[i], long_order, short_order,False, holdings, notional)
        if P[i] < np.mean(P[i:i-10]) - np.std(P[i:i-10]):
            if net_position(long_order, short_order) + \
                    order(False, bid[i], ask[i], long_order, short_order,True, holdings, notional)> - position_limit:
                order(False, bid[i], ask[i], long_order, short_order,False, holdings, notional)
    return PnL




# class Backtesting:
#
#     def __init__(self, list_bid, list_ask, list_transit):
#         self.long_order = []
#         self.short_order = []
#         self.order_list = self.long_order - self.short_order    ## __sub__ is not defined for class list
#         self.transit = list_transit
#         self.bid = list_bid		## list of bid ask price
#         self.ask = list_ask
#         self.position_limit = 100
#
#     def net_position(self):
#         return np.sum(self.order_list)
#
#     def order(self,long_short, i, number_of_currency = 1):   ## can i in main() pass to function?
#         if long_short == True:
#             self.long_order.append(number_of_currency*self.ask[i])
#         else:
#             self.short_order.append(number_of_currency*self.bid[i])
#
#     def main(self):
#         for i in range(10, len(P)-10):
#             assert -position_limit < self.net_position() < position_limit
#             if P[i] > np.mean(P[i-10:i]) + np.std(P[i-10:i]):
#                 if self.net_position() + self.order(long_short=True) < self.position_limit:
#                     self.order(long_short= True,i=i)
#             if P[i] < np.mean(P[i:i-10]) - np.std(P[i:i-10]):
#                 if self.net_position() - self.order(long_short=False)> - self.position_limit:
#                     self.order(long_short= False,i=i)




# [Backtesting(list_bid[x],list_ask[y],list_transit[z]) for x,y,z, in]