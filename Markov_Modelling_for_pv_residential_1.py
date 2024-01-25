import pandas as pd
import numpy as np

######### generate the transition matrix ###########

file_path = './data_generated/residential_power/pv_residential_1.csv'
df = pd.read_csv(file_path)

initial_hour = 12  # initial hour

initial_states = df[df['Hour'] == initial_hour]['pv_states']
print(initial_states)
next_states = df[df['Hour'] == (initial_hour + 1)]['pv_states']
print(next_states)


unique_initial_states = df[df['Hour'] == initial_hour]['pv_states'].unique()
print(unique_initial_states)
unique_next_states = df[df['Hour'] == (initial_hour + 1)]['pv_states'].unique()
print(unique_next_states)

transition_matrix = np.zeros((len(unique_initial_states), len(unique_next_states)))

# statistics the transition times
for i in initial_states.index:
    current_state = initial_states[i]
    next_state = next_states[i + 1]
    current_state_index = np.where(unique_initial_states == current_state)[0][0]
    next_state_index = np.where(unique_next_states == next_state)[0][0]
    transition_matrix[current_state_index, next_state_index] += 1

# normalize the transition matrix, in order to have the possibility of each transition
# for position (n,m), the value is the possibility of state n of initial states to state m of next states, index from 0
transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

# print the states and matrix
print("Unique States:", unique_next_states)
print("Transition Matrix:")
print(transition_matrix)

#########      generation completed      ###########