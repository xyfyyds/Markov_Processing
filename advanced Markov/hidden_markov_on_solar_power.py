import pandas as pd
import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt

file_path = '../data_generated/weather/solar_power_states.csv'
df = pd.read_csv(file_path)

initial_hour_list = []
for h in range(4,18):
    initial_hour_list.append(h)
print(initial_hour_list)

unique_states = df['solar_power_states'].unique()
num_states = len(unique_states)

transition_matrix_list = []
transition_matrix_list = pd.Series(transition_matrix_list)


for hour in initial_hour_list:
    initial_hour = hour  # initial hour

    initial_states = df[df['Hour'] == initial_hour]['solar_power_states']
    print(initial_states)
    next_states = df[df['Hour'] == (initial_hour + 1)]['solar_power_states']
    print(next_states)

    transition_matrix = np.zeros((num_states, num_states))

    # statistics the transition times
    for i in initial_states.index:
        current_state = initial_states[i]
        next_state = next_states[i + 1]
        current_state_index = np.where(unique_states == current_state)[0][0]
        next_state_index = np.where(unique_states == next_state)[0][0]
        transition_matrix[current_state_index, next_state_index] += 1

    for i in range(len(transition_matrix)):
        if np.sum(transition_matrix[i]) == 0:
            transition_matrix[i][i] = num_states

    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)



    transition_matrix_list[hour] = transition_matrix

    print("Transition Matrix:")
    print(transition_matrix)

print(transition_matrix_list)

observations = df['pv_states']
unique_observations = np.unique(observations)
num_observations = len(unique_observations)

observations_matrix = np.zeros((num_states, num_observations))

for temp in unique_states:
    temp_data = df[df['solar_power_states'] == temp]
    for i in range(len(unique_observations)):
        observations_matrix[np.where(unique_states == temp)[0][0], i] = len(temp_data[temp_data['pv_states'] == unique_observations[i]])

observations_matrix = observations_matrix / np.sum(observations_matrix, axis=1, keepdims=True)
print("Observations Matrix:")
print(observations_matrix)
mask = (observations_matrix != 0) & (observations_matrix != 1)
print(np.count_nonzero(mask))


pi = transition_matrix_list[10][np.where(unique_states == 325)[0][0]]
print(pi)


# # HMM model by python package hmmlearn
# model = hmm.CategoricalHMM(n_components=num_states)
# model.startprob_ = pi
# model.transmat_ = transition_matrix_list[11]
# model.emissionprob_ = observations_matrix
# model.n_features = num_observations
# ob_list = [15]
# print(unique_observations[ob_list[0]])
# prob = model.score([ob_list])
# print(f"log_prob_{ob_list}:", prob)


# forward algorithm
def forward(ob_list):
    alpha = pi * observations_matrix[:, ob_list[0]]
    for obs in ob_list[1:]:
        alpha = np.dot(alpha, transition_matrix) * observations_matrix[:, obs]
    return np.sum(alpha)


def backward(ob_list):
    obs_len = len(ob_list)
    beta = np.ones_like(observations_matrix[:, 0])
    for i in list(range(0, obs_len - 1))[::-1]:
        obs = ob_list[i + 1]
        beta = np.dot(transition_matrix, observations_matrix[:, obs] * beta)
    return np.sum(pi * observations_matrix[:, ob_list[0]] * beta)



pi_list = []
initial_hour = 12
for item in df[df['Hour'] == initial_hour]['solar_power_states'].values:
    pi = transition_matrix_list[initial_hour][np.where(unique_states == item)[0][0]]
    pi_list.append(pi)

possible_observations = []
for i in range(num_observations):
    possible_observations.append([i])

result_list = []

for pi in pi_list:
    max_prob = -np.inf
    max_ob = []
    for ob in possible_observations:
        prob = forward(ob)
        if prob > max_prob:
            max_prob = prob
            max_ob = ob
    result_list.append(unique_observations[max_ob[0]])

real_data = df[df['Hour'] == (initial_hour + 1)]['pv_states'].values

overall_difference = 0
count = 0
print(len(result_list))
for i in range(len(result_list) - 1):
    diff = abs(result_list[i] - real_data[i+1])
    overall_difference += diff
    if diff <= abs(real_data[i] * 0.2):
        count += 1

print("Average Difference: ", overall_difference / len(result_list))
print("Percentage of Difference within 20%: ", str((count / len(result_list)) * 100) + "%" )

plt.figure(figsize=(20, 12))
# Plot the predicted states
plt.plot(result_list, label='simulation')
# Plot the original states
plt.plot(real_data[1:], label='real data')

plt.xlabel('Times of transitions')
plt.ylabel('Power')
plt.title('Comparison')
plt.legend()

plt.show()
