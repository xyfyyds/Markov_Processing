import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

######### generate the transition matrix ###########

file_path = './data_generated/residential_power/pv_residential_1.csv'
df = pd.read_csv(file_path)

initial_hour_list = []
for h in range(3,19):
    initial_hour_list.append(h)
print(initial_hour_list)

transition_matrix_list = []
transition_matrix_list = pd.Series(transition_matrix_list)

for hour in initial_hour_list:
    initial_hour = hour  # initial hour

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

    transition_matrix_list[hour] = transition_matrix
    # print the states and matrix
    print("Unique States:", unique_next_states)
    print("Transition Matrix:")
    print(transition_matrix)

print(transition_matrix_list)

#########      generation completed      ###########

# Above is the transition matrix for each hour, now select the hour 12 as test set.
diff_1by1 = 0
count_1by1 = 0
simulation_results = []
real_data = []
overall_difference = 0
initial_states = df[df['Hour'] == 11]['DE_KN_residential1_pv']
unique_initial_states = df[df['Hour'] == 11]['pv_states'].unique()
print(unique_initial_states)
next_states = df[df['Hour'] == 12]['DE_KN_residential1_pv']
unique_next_states = df[df['Hour'] == 12]['pv_states'].unique()
print(unique_next_states)
for i in range(len(initial_states)):
    initial_data = unique_initial_states[np.abs(unique_initial_states - initial_states.iloc[i]).argmin()]
    print("initial_data: ", initial_data)
    current_state_index = np.where(unique_initial_states == initial_data)[0][0]
    next_state_probs = transition_matrix_list[11][current_state_index, :]
    expectation_next_state = 0
    for j in range(len(next_state_probs)):
        expectation_next_state += next_state_probs[j] * unique_next_states[j]
    # expectation_next_state = unique_next_states[np.random.choice(np.arange(0, len(next_state_probs)), p=next_state_probs)]
    simulation_results.append(expectation_next_state)
    real_data.append(next_states.iloc[i])
    print("expectation_next_state: ", expectation_next_state)
    print("next_states.iloc[" + str(i) + "]: ", next_states.iloc[i])
    diff_1by1 = abs(expectation_next_state - next_states.iloc[i])
    overall_difference += diff_1by1
    if diff_1by1 <= abs(next_states.iloc[i] * 0.2):
        count_1by1 += 1
print("Percentage of Difference within 20% predicted one by one: ", str((count_1by1 / len(initial_states)) * 100) + "%")
print("Overall Difference: ", overall_difference/len(initial_states))

plt.figure(figsize=(20, 12))
# Plot the predicted states
plt.plot(simulation_results, label='simulation')
# Plot the original states
plt.plot(real_data, label='real data')

plt.xlabel('Times of transitions')
plt.ylabel('Price')
plt.title('Comparison')
plt.legend()

plt.show()