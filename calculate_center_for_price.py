import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
file_path = './data_generated/price/price_of_DE_LU_cleaned.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path, nrows=15000)

# 提取第三列数据
feature_column = 2  # 假设第三列索引为2，Python中索引从0开始
features = data.iloc[:, feature_column]

# 将数据转换为二维数组形式
X = features.values.reshape(-1, 1)

# 设置聚类数
k = 3600

# 使用K均值聚类
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# 获取聚类结果（质心对应的标签）
labels = kmeans.predict(X)
print(labels)

# 将质心的值添加到原始数据的第四列
data['Centroid_Value'] = np.zeros(len(data))

# 计算每个数据点最接近的质心的值
for i in range(len(data)):
    index = labels[i]
    data.at[i, 'Centroid_Value'] = kmeans.cluster_centers_[index][0]

# 打印质心坐标
centroids = kmeans.cluster_centers_
print("Centroids:")
for centroid in centroids:
    print(centroid)


# 绘制原始数据点和质心
plt.figure(figsize=(40, 24))
plt.scatter(data.iloc[:, feature_column], [0]*len(data), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

# 将结果存回CSV文件
output_file_path = './data_generated/price/cluster_data.csv'  # 替换为输出的CSV文件路径
data.to_csv(output_file_path, index=False)
