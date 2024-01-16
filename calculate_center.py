import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
file_path = './data_generated/price_of_DE_LU_cleaned.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 提取第三列数据
feature_column = 2  # 假设第三列索引为2，Python中索引从0开始
features = data.iloc[:, feature_column]

# 将数据转换为二维数组形式
X = features.values.reshape(-1, 1)

# 设置聚类数（可以根据前面提到的选择聚类数的方法进行选择）
k = 90

# 使用K均值聚类
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# 获取聚类中心
centroids = kmeans.cluster_centers_
for centroid in centroids:
    print(centroid)

# 获取每个数据点的标签（簇索引）
labels = kmeans.labels_

# 绘制数据点和质心
plt.figure(figsize=(40, 24))
plt.scatter(features, [0]*len(features), c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red')
plt.title('K-Means Clustering')
plt.show()