import numpy as np
import pandas as pd
from hmmlearn import hmm
from matplotlib import pyplot as plt

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
pi = transition_matrix[np.where(unique_states == 1.9)[0][0]]
print(pi)


# # HMM model by python package hmmlearn
# model = hmm.CategoricalHMM(n_components=num_states)
# model.startprob_ = pi
# model.transmat_ = transition_matrix
# model.emissionprob_ = observations_matrix
# model.n_features = num_observations
# ob_list = [[2, 3], [2, 56], [12,2]]
# for i in ob_list:
#     prob = model.score([i])
#     print(f"log_prob_{i}:", prob)


def forward(obs_seq):
    alpha = pi * observations_matrix[:, obs_seq[0]]
    for obs in obs_seq[1:]:
        alpha = np.dot(alpha, transition_matrix) * observations_matrix[:, obs]
    return np.sum(alpha)


def backward(obs_seq):
    obs_len = len(obs_seq)
    beta = np.ones_like(observations_matrix[:, 0])
    for i in list(range(0, obs_len - 1))[::-1]:
        obs = obs_seq[i + 1]
        beta = np.dot(transition_matrix, observations_matrix[:, obs] * beta)
    return np.sum(pi * observations_matrix[:, obs_seq[0]] * beta)

# res_forward = forward(obs_seq)
# print(f"res_forward={res_forward}")
# res_backward = backward(obs_seq)
# print(f"res_backward={res_backward}")

# possible_observations = []
# for i in range(num_observations):
#     for j in range(num_observations):
#         for k in range(num_observations):
#             possible_observations.append([i, j, k])

# max_prob = -np.inf
# max_ob = []
# for ob in possible_observations:
#     prob = model.score([ob])
#     if prob > max_prob:
#         max_prob = prob
#         max_ob = ob
#
# print(f"max_prob={max_prob}, max_ob={max_ob}")
# for item in max_ob:
#     print(unique_observations[item])

pi_list = []
for item in states[:1000]:
    pi = transition_matrix[np.where(unique_states == item)[0][0]]
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

real_data = data['price_state'][:1001]

overall_difference = 0
count = 0
for i in range(len(result_list)):
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
plt.plot(real_data[1:], label='real price')

plt.xlabel('Times of transitions')
plt.ylabel('Price')
plt.title('Comparison')
plt.legend()

plt.show()
