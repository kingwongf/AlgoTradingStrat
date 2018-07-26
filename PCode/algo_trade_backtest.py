import numpy as np

transition matrix = read_.h5()

list_price = read_.price()

P00, P01, P10, P11 = transition matrix

## 1: higher mu, 0: lower mu
Class Backtesting():
	def __init__:
		self.long_order = []
		self.short_order = []
		self.order_list = long_order - short_order
		self.transit = transition matrix
		self.bid = list_bid		## list of bid ask price
		self.ask = list_ask
		self.position_limit = 100

	def net_position(self):
		return np.sum(self.order_list)

	def order(long, number of currency = 1): 
		if long == True:
			self.long_order.append(number of currency*self.ask[i])
		else:
			self.short_order.append(number of currency*self.bid[i])


	def main():
		for i in len(range(P00)):
			assert -position_limit < self.net_position() < position_limit
			if P01[i] > np.mean(P[i:i-10]) + np.std(P[i:i-10]):
				if self.net_position() + self.order_size(long=True) < self.position_limit :
					self.order(long = True)
			if P10[i] < np.mean(P[i:i-10]) - np.std(P[i:i-10]):
				if self.net_position() - self.order_size(long=False)> - self.position_limit:
					self.order(long = False)


number_of_currency = 1
position_limit = 100