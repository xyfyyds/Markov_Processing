import numpy as np
import pandas as pd

file_path = '../data_generated/weather/temperature_price.csv'
data = pd.read_csv(file_path)
prices = data.iloc[:, 1]
print(prices)

max_price = prices.max()
min_price = prices.min()
print(f'Maximum price: {max_price}\n'f'Minimum price: {min_price}\n')

interval_width = 0.5
intervals = np.arange(min_price, max_price + interval_width, interval_width)
print(intervals)

data['price_state'] = pd.cut(prices, bins=intervals, labels=intervals[:-1] + interval_width / 2)
data['price_state'] = data['price_state'].apply(lambda x: round(x, 2))
print(data['price_state'].unique())
print(len(data['price_state'].unique()))

temperatures = data.iloc[:, 2]
max_temperature = temperatures.max()
min_temperature = temperatures.min()
print(f'Maximum price: {max_temperature}\n'f'Minimum price: {min_temperature}\n')
print(len(temperatures.unique()))

data.to_csv('../data_generated/weather/temperature_price_states.csv', index=False)