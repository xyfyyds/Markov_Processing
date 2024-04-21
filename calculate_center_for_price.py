import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

'''
    calculate the center of the price data, that is to learn the states
'''

file_path = './data_generated/price/price_of_DE_LU_cleaned.csv'
data = pd.read_csv(file_path, nrows=15000)

feature_column = 2
features = data.iloc[:, feature_column]

X = features.values.reshape(-1, 1)

k = 100

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

labels = kmeans.predict(X)
print(labels)

data['Centroid_Value'] = np.zeros(len(data))

for i in range(len(data)):
    index = labels[i]
    data.at[i, 'Centroid_Value'] = kmeans.cluster_centers_[index][0]

centroids = kmeans.cluster_centers_
print("Centroids:")
for centroid in centroids:
    print(centroid)

plt.figure(figsize=(40, 24))
plt.scatter(data.iloc[:, feature_column], [0]*len(data), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

output_file_path = './data_generated/price/cluster_data.csv'  # 替换为输出的CSV文件路径
data.to_csv(output_file_path, index=False)
