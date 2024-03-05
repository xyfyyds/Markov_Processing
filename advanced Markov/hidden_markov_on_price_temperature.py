import numpy as np
import pandas as pd

# 从CSV文件中读取数据
file_path = '../data_generated/weather/temperature_price_states.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

states = data['temperature']
unique_states = np.unique(states)

# set up the transition matrix, initially all zeros
num_states = len(unique_states)
print(num_states)
transition_matrix = np.zeros((num_states, num_states))

# statistics the transition times
for i in range(len(states) - 1):
    current_state = states.iloc[i]
    next_state = states.iloc[i + 1]
    current_state_index = np.where(unique_states == current_state)[0][0]
    next_state_index = np.where(unique_states == next_state)[0][0]
    transition_matrix[current_state_index, next_state_index] += 1

# normalize the transition matrix, in order to have the possibility of each transition
# for position (n,m), the value is the possibility of state n to state m, index from 0
# transition_matrix is the states transition matrix A
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

# print the states and matrix
print("Unique States:", unique_states)
print("Transition Matrix:")
print(transition_matrix)
mask = (transition_matrix != 0) & (transition_matrix != 1)
print(np.count_nonzero(mask))

observations = data['price_state']
unique_observations = np.unique(observations)
print(unique_observations[114])
print(unique_observations[117])
print(unique_observations[120])
print(unique_observations[121])
num_observations = len(unique_observations)

observations_matrix = np.zeros((num_states, num_observations))

for temp in unique_states:
    temp_data = data[data['temperature'] == temp]
    for i in range(len(unique_observations)):
        observations_matrix[np.where(unique_states == temp)[0][0], i] = len(temp_data[temp_data['price_state'] == unique_observations[i]])

observations_matrix = observations_matrix / np.sum(observations_matrix, axis=1, keepdims=True)
print("Observations Matrix:")
print(observations_matrix)
mask = (observations_matrix != 0) & (observations_matrix != 1)
print(np.count_nonzero(mask))

# select the initial state probability vector 2020.9.28 23:00
pi = transition_matrix[np.where(unique_states == 9.4)[0][0]]
print(pi)
