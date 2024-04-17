import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

### states calculation
'''
file_path = './data_generated/residential_power/power_consumption_residential_1.csv'
df = pd.read_csv(file_path)

# transform the time column to datetime type
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# obtain the hours, create a new column
df['Hour'] = df['utc_timestamp'].dt.hour

hourly_data = df.groupby('Hour')['total_consumption']

hourly_counts = df.groupby('Hour')['total_consumption'].count()
print("hourly_count: " + str(hourly_counts))

data_of_each_hour = hourly_data.apply(list)

data = np.array(data_of_each_hour[6])
print(data)
X = np.array(data).reshape(-1, 1)

for i in range(10):
    kmeans = KMeans(n_clusters=120)
    kmeans.fit(X)
    labels = kmeans.labels_
    score = silhouette_score(X, labels)
    print(score)

k_list = [5, 5, 10, 200, 100, 100]
# add a new column to store the states
df['consumption_states_per_hour'] = np.zeros(len(df))

for per_hour in range(24):
    print("per_hour: " + str(per_hour))

    data = np.array(data_of_each_hour[per_hour])

    X = np.array(data).reshape(-1, 1)

    # k-means
    if per_hour < 6:
        k = k_list[per_hour]
    else:
        k = 100
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    # obtain the labels
    labels = kmeans.predict(X)

    # calculate the nearest centroid for each data point
    count = 0
    for i in range(len(df)):
        if df.at[i, 'Hour'] == per_hour:
            index = labels[count]
            count += 1
            df.at[i, 'consumption_states_per_hour'] = kmeans.cluster_centers_[index][0]

    # print the centers
    centroids = kmeans.cluster_centers_
    print("Centroids:")
    for centroid in centroids:
        print(centroid)

# store useful columns to a new csv file
output_file_path = './data_generated/residential_power/per_pc_residential_1.csv'
load_data = df[['utc_timestamp', 'total_consumption', 'Hour', 'consumption_states_per_hour']]
load_data.to_csv(output_file_path, index=False)
'''


### matrix generation
file_path = './data_generated/residential_power/per_pc_residential_1.csv'
df = pd.read_csv(file_path)

transition_matrix_list = []
transition_matrix_list = pd.Series(transition_matrix_list)

for hour in range(24):
    initial_hour = hour  # initial hour
    print("initial_hour: " + str(initial_hour))

    initial_states = df[df['Hour'] == initial_hour]['consumption_states_per_hour']
    print(initial_states)
    next_states = df[df['Hour'] == ((initial_hour + 1) % 24) ]['consumption_states_per_hour']
    print(next_states)


    unique_initial_states = df[df['Hour'] == initial_hour]['consumption_states_per_hour'].unique()
    print(unique_initial_states)
    unique_next_states = df[df['Hour'] == ((initial_hour + 1) % 24) ]['consumption_states_per_hour'].unique()
    print(unique_next_states)

    transition_matrix = np.zeros((len(unique_initial_states), len(unique_next_states)))

    # statistics the transition times
    for i in initial_states.index:
        current_state = initial_states[i]
        if i < 15871:
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


# Above is the transition matrix for each hour, now select the hour 3/4 as test set.
diff_1by1 = 0
count_1by1 = 0
simulation_results = []
real_data = []
overall_difference = 0
initial_states = df[df['Hour'] == 3]['total_consumption']
unique_initial_states = df[df['Hour'] == 3]['consumption_states_per_hour'].unique()
print(unique_initial_states)
next_states = df[df['Hour'] == 4]['total_consumption']
unique_next_states = df[df['Hour'] == 4]['consumption_states_per_hour'].unique()
print(unique_next_states)
for i in range(len(initial_states)):
    initial_data = unique_initial_states[np.abs(unique_initial_states - initial_states.iloc[i]).argmin()]
    print("initial_data: ", initial_data)
    current_state_index = np.where(unique_initial_states == initial_data)[0][0]
    next_state_probs = transition_matrix_list[3][current_state_index, :]
    expectation_next_state = 0
    for j in range(len(next_state_probs)):
        expectation_next_state += next_state_probs[j] * unique_next_states[j]
    expectation_next_state = unique_next_states[np.random.choice(np.arange(0, len(next_state_probs)), p=next_state_probs)]
    simulation_results.append(expectation_next_state)
    real_data.append(next_states.iloc[i])
    print("expectation_next_state: ", expectation_next_state)
    print("next_states.iloc[" + str(i) + "]: ", next_states.iloc[i])
    diff_1by1 = abs(expectation_next_state - next_states.iloc[i])
    overall_difference += diff_1by1
    if diff_1by1 <= abs(next_states.iloc[i] * 0.2):
        count_1by1 += 1
print("Percentage of Difference within 20% predicted one by one: ", str((count_1by1 / len(initial_states)) * 100) + "%")
print("Overall Difference: ", overall_difference / len(initial_states))

plt.figure(figsize=(20, 12))
# Plot the predicted states
plt.plot(simulation_results[:200], label='simulation')
# Plot the original states
plt.plot(real_data[:200], label='real demand')

plt.xlabel('Times of transitions')
plt.ylabel('Price')
plt.title('Comparison')
plt.legend()

plt.show()
