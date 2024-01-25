import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture


file_path = './data_generated/residential_power/power_of_residential_1_changes.csv'
df = pd.read_csv(file_path)

# transform the time column to datetime type
df['Time'] = pd.to_datetime(df['utc_timestamp'])

# obtain the hours, create a new column
df['Hour'] = df['Time'].dt.hour

hourly_data = df[(df['DE_KN_residential1_pv'] != 0) & (~df['DE_KN_residential1_pv'].isna())].groupby('Hour')['DE_KN_residential1_pv']

hourly_counts = df[(df['DE_KN_residential1_pv'] != 0) & (df['DE_KN_residential1_pv'] is not None)].groupby('Hour')['DE_KN_residential1_pv'].count()
print(hourly_counts)

data_of_each_hour = hourly_data.apply(list)

data = np.array(data_of_each_hour[12])
print(data)

X = np.array(data).reshape(-1, 1)

######## GMM model ########
n_components = 60  # 设置聚类的数量
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)

# obtain the means and stds
means = gmm.means_
stds = np.sqrt(gmm.covariances_)
print("Means:", means)
print("Stds:", stds)

# calculate scores
labels = gmm.fit_predict(X)
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)
ch_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", ch_score)

# plot
x_vals = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
log_probs = gmm.score_samples(x_vals)
y_vals = np.exp(log_probs)  # 使用 score_samples 得到每个点的概率密度值

for i in range(n_components):
    plt.plot(x_vals, y_vals)

plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Gaussian Mixture Model Fit to One-Dimensional Data')
plt.legend()
plt.show()

###########################################

######## MeanShift model ########
bandwidth = estimate_bandwidth(X, quantile=0.02, n_samples=1000)

mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(X)

# obtain the centers
centroids = mean_shift.cluster_centers_
print("Centroids:", centroids)

# obtain the labels
labels = mean_shift.labels_

# calculate scores
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

# plot
plt.figure(figsize=(10, 6))
for i in np.unique(labels):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points, [i] * len(cluster_points), label=f'Cluster {i}')

plt.scatter(centroids, np.arange(len(centroids)), marker='x', color='red', s=100, label='Centroids')

plt.xlabel('Data')
plt.ylabel('Cluster')
plt.title('MeanShift Clustering')
plt.legend()
plt.show()

###########################################

######## K-Means model ########

# K value
k = 60

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

labels = kmeans.labels_
silhouette_avg = silhouette_score(X, labels)
print("Silhouette Score:", silhouette_avg)
ch_score = calinski_harabasz_score(X, labels)
print("Calinski-Harabasz Score:", ch_score)

# print centers
centroids = kmeans.cluster_centers_
print("Centroids:")
for centroid in centroids:
    print(centroid)

# plot
plt.figure(figsize=(40, 24))
plt.scatter(data, [0]*len(data), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
