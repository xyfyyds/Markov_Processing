import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

'''
    learn the number of clusters for the price data
'''

file_path = './data_generated/price/price_of_DE_LU_cleaned.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path, nrows=15000)

feature_column = 2  # 假设第三列索引为2，Python中索引从0开始
features = data.iloc[:, feature_column]

X = features.values.reshape(-1, 1)

inertia_values = []
k_values = range(1, 101)

# for k in k_values:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     inertia_values.append(kmeans.inertia_)
#
# plt.figure(figsize=(40, 24))
# plt.plot(k_values, inertia_values, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()

# 通过轮廓系数选择最优聚类数
silhouette_scores = []


for i in range(1,10):
    kmeans = KMeans(n_clusters=3600)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    print(score)

# for k in k_values[1:]:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     labels = kmeans.labels_
#     score = silhouette_score(X, labels)
#     silhouette_scores.append(score)
#
# plt.figure(figsize=(40, 24))
# plt.plot(k_values[1:], silhouette_scores, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score Method')
# plt.show()


