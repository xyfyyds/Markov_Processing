import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

# 读取CSV文件，假设时间列为第一列，第六列为数据列
file_path = './data_generated/residential_power/power_of_residential_1_changes.csv'
df = pd.read_csv(file_path)

# 将时间列转换为 datetime 类型
df['Time'] = pd.to_datetime(df['utc_timestamp'])

# 提取小时信息，创建新列
df['Hour'] = df['Time'].dt.hour

hourly_data = df[(df['DE_KN_residential1_pv'] != 0) & (~df['DE_KN_residential1_pv'].isna())].groupby('Hour')['DE_KN_residential1_pv']

hourly_counts = df[(df['DE_KN_residential1_pv'] != 0) & (df['DE_KN_residential1_pv'] is not None)].groupby('Hour')['DE_KN_residential1_pv'].count()
print(hourly_counts)

data_of_each_hour = hourly_data.apply(list)

data = np.array(data_of_each_hour[12])
print(data)

X = np.array(data).reshape(-1, 1)

######## 定义 GMM 模型并拟合数据 ########
n_components = 60  # 设置聚类的数量
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)

# 获取每个组件的均值和标准差
means = gmm.means_
stds = np.sqrt(gmm.covariances_)
print("Means:", means)
print("Stds:", stds)

# 计算分数
labels = gmm.fit_predict(X)
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)
ch_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", ch_score)

# 生成用于绘图的数据点
x_vals = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
log_probs = gmm.score_samples(x_vals)
y_vals = np.exp(log_probs)  # 使用 score_samples 得到每个点的概率密度值

# 绘制 GMM 模型拟合的概率密度函数
for i in range(n_components):
    plt.plot(x_vals, y_vals)

plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Gaussian Mixture Model Fit to One-Dimensional Data')
plt.legend()
plt.show()

###########################################

######## 定义 MeanShift 模型并拟合数据 ########
bandwidth = estimate_bandwidth(X, quantile=0.02, n_samples=1000)
# 使用 MeanShift 进行聚类
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(X)

# 获取聚类中心
centroids = mean_shift.cluster_centers_
print("Centroids:", centroids)

# 获取每个数据点的标签
labels = mean_shift.labels_

# 计算分数
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)
ch_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", ch_score)

cluster_ranges = []
for i, centroid in enumerate(centroids):
    cluster_points = X[labels == i]
    cluster_range = (cluster_points.min(), cluster_points.max())
    cluster_ranges.append(cluster_range)
    print(f"Cluster {i}: {cluster_range}")

# 绘制聚类结果
plt.figure(figsize=(10, 6))
for i in np.unique(labels):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points, [i] * len(cluster_points), label=f'Cluster {i}')

# 绘制聚类中心
plt.scatter(centroids, np.arange(len(centroids)), marker='x', color='red', s=100, label='Centroids')

plt.xlabel('Data')
plt.ylabel('Cluster')
plt.title('MeanShift Clustering')
plt.legend()
plt.show()

###########################################

######## 定义 K-Means 模型并拟合数据 ########

# 设置聚类数
k = 60

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

labels = kmeans.labels_
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)
ch_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", ch_score)

# 打印质心坐标
centroids = kmeans.cluster_centers_
print("Centroids:")
for centroid in centroids:
    print(centroid)

# 绘制原始数据点和质心
plt.figure(figsize=(40, 24))
plt.scatter(data, [0]*len(data), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
