import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

### model for the overall consumption

### center number calculation, do not run this part unless showing the result again ###
'''
file_path = './data_generated/residential_power/power_consumption_residential_1.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

feature_column = 5
features = data.iloc[:, feature_column]

print(features)

X = features.values.reshape(-1, 1)

silhouette_scores = []
k_values = range(1, 101)

for i in range(1,10):
    kmeans = KMeans(n_clusters=1000)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    print(score)
'''
### center number initially set as 1000, when the score is just over 0.71 ###

### state/center calculation, do not run this part unless regenerating the states ###
'''
file_path = './data_generated/residential_power/power_consumption_residential_1.csv'
data = pd.read_csv(file_path)

feature_column = 5
features = data.iloc[:, feature_column]
X = features.values.reshape(-1, 1)

print(features)

k = 1000
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.predict(X)

data['consumption_states'] = np.zeros(len(data))

# 计算每个数据点最接近的质心的值
for i in range(len(data)):
    index = labels[i]
    data.at[i, 'consumption_states'] = kmeans.cluster_centers_[index][0]

centroids = kmeans.cluster_centers_
print("Centroids:")
for centroid in centroids:
    print(centroid)

plt.figure(figsize=(40, 24))
plt.scatter(data.iloc[:, feature_column], [0]*len(data), c=labels, cmap='viridis', alpha=0.5, label='Data Points')
plt.scatter(centroids, [0]*len(centroids), marker='X', s=200, color='red', label='consumption_states')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

output_file_path = './data_generated/residential_power/power_consumption_residential_1.csv'  # 替换为输出的CSV文件路径
data.to_csv(output_file_path, index=False)
'''
### start modelling

### generate the transition matrix ###
file_path = './data_generated/residential_power/power_consumption_residential_1.csv'
data = pd.read_csv(file_path)

state_column = 6
states = data.iloc[:, state_column]

print(states[0])

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

### predict the future states ###

# Number of steps to predict
num_steps = 15000

Initial_value = states[0]
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

# print("Predicted States:" + str(predicted_states))

# plt.figure(figsize=(20, 12))
# # Plot the predicted states
# plt.plot(predicted_states, label='predicted price')

# Read the original states
df = pd.read_csv('./data_generated/residential_power/power_consumption_residential_1.csv')

column_index = 5
data_to_plot = df.iloc[0:15000, column_index]
print("Initial data:" + str(data_to_plot.values[0]))

# the average absolute difference between the predicted states and the original states
diff_abs = 0
for i in range(num_steps):
    diff_abs += abs(predicted_states[i] - data_to_plot.values[i])
print("Average Absolute Difference: ", diff_abs / num_steps)

# the percentage of difference within 20% between the predicted states and the original states predicted one by one
diff_1by1 = 0
count_1by1 = 0
simulation_results = []
real_data = []
overall_difference = 0
for i in range(len(data_to_plot) - 1):
    initial_data = unique_states[np.abs(unique_states - data_to_plot.values[i]).argmin()]
    current_state_index = np.where(unique_states == initial_data)[0][0]
    next_state_probs = transition_matrix[current_state_index, :]
    next_state = np.random.choice(np.arange(0, len(next_state_probs)), p=next_state_probs)
    simulation_results.append(unique_states[next_state])
    real_data.append(data_to_plot.values[i+1])
    diff_1by1 = abs(unique_states[next_state] - data_to_plot.values[i+1])
    overall_difference += diff_1by1
    if diff_1by1 <= abs(data_to_plot.values[i+1] * 0.2):
        count_1by1 += 1
print("Percentage of Difference within 20% predicted one by one: ", str((count_1by1 / len(data_to_plot)) * 100) + "%")
print("Overall Difference: ", overall_difference/len(data_to_plot))

plt.figure(figsize=(20, 12))
# Plot the predicted states
plt.plot(simulation_results[:100], label='simulation')
# Plot the original states
plt.plot(real_data[:100], label='real demand')

plt.xlabel('Times of transitions')
plt.ylabel('Price')
plt.title('Comparison')
plt.legend()

plt.show()

# Plot the original states
# plt.plot(range(0, 15000), data_to_plot, label='real price')
#
# plt.xlabel('Times of changes of states')
# plt.ylabel('Price')
# plt.title('Comparison of Data')
# plt.legend()
#
# plt.show()