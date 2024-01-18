import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

######### generate the transition matrix ###########

# read from csv file
file_path = './data_generated/cluster_data.csv'
data = pd.read_csv(file_path)

# select the 4th column as the input data
state_column = 3  # 0,1,2,3 index from 0
states = data.iloc[:, state_column]

# obtain the unique states, that is unique values in the 4th column
unique_states = np.unique(states)

# set up the transition matrix, initially all zeros
num_states = len(unique_states)
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
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

# print the states and matrix
print("Unique States:", unique_states)
print("Transition Matrix:")
print(transition_matrix)

#########      generation completed      ###########


######### predict the future states ###########

# Number of steps to predict
num_steps = 500

# Initial state (choose one of the unique states)
initial_state = 82  # For example, choose the second state

# Store the predicted states
predicted_states = [initial_state]

# Predict future states
for _ in range(num_steps):
    current_state_index = predicted_states[-1]
    next_state_probs = transition_matrix[current_state_index, :]
    next_state = np.random.choice(np.arange(0, len(next_state_probs)), p=next_state_probs)
    predicted_states.append(next_state)

for i in range(len(predicted_states)):
    predicted_states[i] = unique_states[predicted_states[i]]

print("Predicted States:" + str(predicted_states))

plt.figure(figsize=(10, 5))
# Plot the predicted states
plt.plot(predicted_states, label='predicted data')

# Plot the original states
df = pd.read_csv(file_path)

column_index = 3
data_to_plot = df.iloc[500:1001, column_index]

plt.plot(range(0, 501), data_to_plot, label='original data')

# 添加标签和标题
plt.xlabel('Row Index')
plt.ylabel('Data Value')
plt.title('Comparison of Two Data Ranges in the 4th Column')
plt.legend()  # 显示图例

plt.show()
