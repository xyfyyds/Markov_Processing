import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

file_path = './data_generated/residential_power/power_of_residential_1_changes.csv'
df = pd.read_csv(file_path)

# transform the time column to datetime type
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# obtain the hours, create a new column
df['Hour'] = df['utc_timestamp'].dt.hour

hourly_data = df[(df['DE_KN_residential1_pv'] != 0) & (~df['DE_KN_residential1_pv'].isna())].groupby('Hour')['DE_KN_residential1_pv']

hourly_counts = df[(df['DE_KN_residential1_pv'] != 0) & (~df['DE_KN_residential1_pv'].isna())].groupby('Hour')['DE_KN_residential1_pv'].count()
print("hourly_count: " + str(hourly_counts))

data_of_each_hour = hourly_data.apply(list)

# add a new column to store the states
df['pv_states'] = np.zeros(len(df))
#k value for each hour
k_list = [5, 20, 30, 40, 60, 60, 60, 60, 60, 60, 60, 60, 60, 40, 30, 20, 4]

print(type(df.at[0, 'DE_KN_residential1_pv']))

for per_hour in range(3, 20):
    print("per_hour: " + str(per_hour))

    data = np.array(data_of_each_hour[per_hour])
    print("data: " + str(data))

    X = np.array(data).reshape(-1, 1)

    # k-means
    k = k_list[per_hour - 3]

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    # obtain the labels
    labels = kmeans.predict(X)
    print("labels: " + str(labels))

    # calculate the nearest centroid for each data point
    count = 0
    for i in range(len(df)):
        if (df.at[i, 'Hour'] == per_hour) & (df.at[i, 'DE_KN_residential1_pv'] != 0) & (~np.isnan(df.at[i, 'DE_KN_residential1_pv'])):
            index = labels[count]
            count += 1
            df.at[i, 'pv_states'] = kmeans.cluster_centers_[index][0]

    # print the centers
    centroids = kmeans.cluster_centers_
    print("Centroids:")
    for centroid in centroids:
        print(centroid)


# store useful columns to a new csv file
output_file_path = './data_generated/residential_power/pv_residential_1.csv'
load_data = df[['utc_timestamp', 'DE_KN_residential1_pv', 'Hour', 'pv_states']]
load_data.to_csv(output_file_path, index=False)