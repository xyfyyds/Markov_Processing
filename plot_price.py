import pandas as pd
import matplotlib.pyplot as plt

file_path = './data_generated/price/price_of_DE_LU_cleaned.csv'
data = pd.read_csv(file_path)

data['Time'] = pd.to_datetime(data.iloc[:, 0])

x_axis = data['Time']
y_axis = data.iloc[:, 2]

plt.figure(figsize=(20, 12))

plt.plot(x_axis, y_axis, marker='o', linestyle='-')

plt.title('Price by time')
plt.xlabel('time')
plt.ylabel('price')

plt.xticks(rotation=45)

plt.show()
