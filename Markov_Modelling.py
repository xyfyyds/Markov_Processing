import pandas as pd
import numpy as np

# 从CSV文件中读取数据
file_path = './data_generated/cluster_data.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 提取第四列数据
state_column = 3  # 假设第四列索引为3，Python中索引从0开始
states = data.iloc[:, state_column]

# 获取所有唯一的状态
unique_states = np.unique(states)

# 构建转移矩阵
num_states = len(unique_states)
transition_matrix = np.zeros((num_states, num_states))

# 统计每个状态和其对应的下一个状态的出现次数
for i in range(len(states) - 1):
    current_state = states.iloc[i]
    next_state = states.iloc[i + 1]
    current_state_index = np.where(unique_states == current_state)[0][0]
    next_state_index = np.where(unique_states == next_state)[0][0]
    transition_matrix[current_state_index, next_state_index] += 1

# 将转移矩阵的每一行归一化，得到概率矩阵
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

# 输出唯一状态和转移矩阵
print("Unique States:", unique_states)
print("Transition Matrix:")
print(transition_matrix)
