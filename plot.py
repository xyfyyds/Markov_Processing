import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
file_path = './data_generated/price_of_DE_LU_cleaned.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 将第一列时间列转换为datetime类型
data['Time'] = pd.to_datetime(data.iloc[:, 0])  # 假设第一列是时间列

# 使用时间列作为横轴，第三列作为纵轴
x_axis = data['Time']
y_axis = data.iloc[:, 2]  # 假设第三列是纵轴

# 设置图形大小
plt.figure(figsize=(20, 12))

# 绘制折线图
plt.plot(x_axis, y_axis, marker='o', linestyle='-')

# 添加标题和标签
plt.title('Price by time')
plt.xlabel('time')
plt.ylabel('price')

# 设置横轴刻度标签
plt.xticks(rotation=45)  # 旋转45度，以免标签重叠

# 显示图形
plt.show()
