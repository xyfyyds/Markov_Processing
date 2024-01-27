import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

######### generate the transition matrix ###########

# read from csv file
file_path = './data_generated/price/cluster_data.csv'
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
num_steps = 2502

Initial_value = 26.53
initial_state = 0
# Initial state (choose one of the unique states)
for n in range(len(unique_states)):
    if unique_states[n] == Initial_value:
        initial_state = n
        break
print("Initial state:", initial_state)
print("Initial value:", unique_states[initial_state])

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

plt.figure(figsize=(20, 12))
# Plot the predicted states
plt.plot(predicted_states, label='predicted price')

# Read the original states
df = pd.read_csv('./data_generated/price/price_of_DE_LU_cleaned.csv')

column_index = 2
data_to_plot = df.iloc[14999:17501, column_index]
print("Initial data:" + str(data_to_plot.values[0]))

### Comparison between predicted states and the original states in several methods of simulation ###

# the average absolute difference between the predicted states and the original states
diff_abs = 0
for i in range(num_steps):
    diff_abs += abs(predicted_states[i] - data_to_plot.values[i])
print("Average Absolute Difference: ", diff_abs / num_steps)

# the average difference between the predicted states and the original states
diff_total = 0
for i in range(num_steps):
    diff_total += (predicted_states[i] - data_to_plot.values[i])
print("Average total Difference: ", diff_total / num_steps)

# the percentage of difference within 20% between the predicted states and the original states predicted together
diff_per = 0
count = 0
for i in range(num_steps):
    diff_per = abs(predicted_states[i] - data_to_plot.values[i])
    if diff_per <= abs(data_to_plot.values[i] * 0.2):
        count += 1
print("Percentage of Difference within 20% predicted once: ", str((count / num_steps) * 100) + "%")

# the percentage of difference within 20% between the predicted states and the original states predicted one by one
diff_1by1 = 0
count_1by1 = 0
for i in range(len(data_to_plot) - 1):
    initial_data = unique_states[np.abs(unique_states - data_to_plot.values[i]).argmin()]
    current_state_index = np.where(unique_states == initial_data)[0][0]
    next_state_probs = transition_matrix[current_state_index, :]
    next_state = np.random.choice(np.arange(0, len(next_state_probs)), p=next_state_probs)
    diff_1by1 = abs(unique_states[next_state] - data_to_plot.values[i+1])
    if diff_1by1 <= abs(data_to_plot.values[i+1] * 0.2):
        count_1by1 += 1
print("Percentage of Difference within 20% predicted one by one: ", str((count_1by1 / len(data_to_plot)) * 100) + "%")

########## Finish Comparison ##########

# Plot the original states
plt.plot(range(0, 2502), data_to_plot, label='real price')

plt.xlabel('Times of changes of states')
plt.ylabel('Price')
plt.title('Comparison of Data after the 15000th row')
plt.legend()

plt.show()
