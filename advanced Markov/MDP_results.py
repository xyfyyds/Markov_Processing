import numpy as np
import pandas as pd

'''
    apply MDP strategy on the data, to learn if cost is reduced
'''

from dateutil.relativedelta import relativedelta

# df1 = pd.read_csv('../data_generated/price/cluster_data.csv')
# df2 = pd.read_csv('../data_generated/weather/solar_power_states.csv')
# df3 = pd.read_csv('../data_generated/residential_power/consumption_data.csv')
#
# df1.rename(columns={'utc_timestamp': 'Time'}, inplace=True)
# df1['Time'] = pd.to_datetime(df1['Time']).dt.tz_localize(None)
#
# df2['Time'] = pd.to_datetime(df2['Time']).apply(lambda x: x + relativedelta(years=3)).dt.tz_localize(None)
# df3['Time'] = pd.to_datetime(df3['Time']).apply(lambda x: x + relativedelta(years=3)).dt.tz_localize(None)
#
#
# merged_df = pd.merge(df1, df2, on='Time', how='inner')
# merged_df = pd.merge(merged_df, df3, on='Time', how='inner')
# merged_df = merged_df[['Time', 'Centroid_Value', 'pv_states', 'con_states']]
# merged_df.rename(columns={'Centroid_Value': 'price_state'}, inplace=True)
#
# print(merged_df)
#
# merged_df.to_csv('../data_generated/residential_power/overall_data.csv', index=False)

def price_TM_generation(file_path, state_column):
    file_path = file_path
    data = pd.read_csv(file_path)

    state_column = state_column
    all_prices = data.iloc[:, state_column]

    # obtain the unique states, that is unique values in the 4th column
    unique_states = np.unique(all_prices)

    # set up the transition matrix, initially all zeros
    num_states = len(unique_states)
    transition_matrix = np.zeros((num_states, num_states))

    # statistics the transition times
    for i in range(len(all_prices) - 1):
        current_state = all_prices.iloc[i]
        next_state = all_prices.iloc[i + 1]
        current_index = np.where(unique_states == current_state)[0][0]
        next_index = np.where(unique_states == next_state)[0][0]
        transition_matrix[current_index, next_index] += 1

    # normalize the transition matrix, in order to have the possibility of each transition
    # for position (n,m), the value is the possibility of state n to state m, index from 0
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

    # print the states and matrix
    print("prices States:", unique_states)
    print(" Prices Transition Matrix:")
    print(transition_matrix)

    return num_states, unique_states, transition_matrix

def solar_power_TMList_generation(filepath):
    file_path = filepath
    df = pd.read_csv(file_path)

    initial_hour_list = []
    for h in range(4, 18):
        initial_hour_list.append(h)
    print(initial_hour_list)

    unique_states = df['pv_states'].unique()
    num_states = len(unique_states)

    transition_matrix_list = []
    transition_matrix_list = pd.Series(transition_matrix_list)

    for hour in initial_hour_list:
        initial_hour = hour  # initial hour

        initial_states = df[df['Hour'] == initial_hour]['pv_states']
        next_states = df[df['Hour'] == (initial_hour + 1)]['pv_states']

        transition_matrix = np.zeros((num_states, num_states))

        # statistics the transition times
        for i in initial_states.index:
            current_state = initial_states[i]
            next_state = next_states[i + 1]
            current_index = np.where(unique_states == current_state)[0][0]
            next_index = np.where(unique_states == next_state)[0][0]
            transition_matrix[current_index, next_index] += 1

        for i in range(len(transition_matrix)):
            if np.sum(transition_matrix[i]) == 0:
                transition_matrix[i][i] = num_states

        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
        transition_matrix_list[hour] = transition_matrix

    print(" Solar Power Transition Matrix List:")
    print(transition_matrix_list)

    return num_states, unique_states, transition_matrix_list

def power_consumption_TMList_generation(filepath):
    file_path = filepath
    df = pd.read_csv(file_path)

    unique_states = df['con_states'].unique()
    num_states = len(unique_states)

    transition_matrix_list = []
    transition_matrix_list = pd.Series(transition_matrix_list)

    for hour in range(24):
        initial_hour = hour  # initial hour

        initial_states = df[df['Hour'] == initial_hour]['con_states']
        next_states = df[df['Hour'] == ((initial_hour + 1) % 24)]['con_states']

        transition_matrix = np.zeros((num_states, num_states))

        # statistics the transition times
        for i in initial_states.index:
            current_state = initial_states[i]
            if i < 15871:
                next_state = next_states[i + 1]
                current_index = np.where(unique_states == current_state)[0][0]
                next_index = np.where(unique_states == next_state)[0][0]
                transition_matrix[current_index, next_index] += 1

        for i in range(len(transition_matrix)):
            if np.sum(transition_matrix[i]) == 0:
                transition_matrix[i][i] = num_states

        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
        transition_matrix_list[hour] = transition_matrix

    print(" Power Consumption Transition Matrix List:")
    print(transition_matrix_list)

    return num_states, unique_states, transition_matrix_list


num_prices_states, prices, price_transition_matrix = price_TM_generation('../data_generated/price/cluster_data.csv', 3)
num_solar_power_states, solar_power, solar_power_transition_matrix_list = solar_power_TMList_generation('../data_generated/weather/solar_power_states.csv')
num_consumption_states, consumption, consumption_transition_matrix_list = power_consumption_TMList_generation('../data_generated/residential_power/consumption_data.csv')

states = [(b, p, q) for b in range(num_prices_states) for p in range(num_consumption_states) for q in range(num_solar_power_states)]  # 状态空间

strategy = pd.read_csv("./strategy_by_Q_reward.csv")
states_data = pd.read_csv("../data_generated/residential_power/overall_data.csv")

prices_list_1 = states_data['price_state']
print(len(prices_list_1))
consumption_list_1 = states_data['con_states']
solar_power_list_1 = states_data['pv_states']

action_list = []
for i in range(len(prices_list_1)):
    price_state = np.where(prices == prices_list_1.iloc[i])[0][0] if prices_list_1.iloc[i] in prices else 40
    consumption_state = np.where(consumption == consumption_list_1[i])[0][0]
    solar_power_state = np.where(solar_power == solar_power_list_1[i])[0][0]

    current_state_index = states.index((price_state, consumption_state, solar_power_state))
    if (strategy['charge'][current_state_index] > strategy['discharge'][current_state_index]) & (strategy['charge'][current_state_index] > strategy['idle'][current_state_index]):
        action_list.append("charge")
    if (strategy['discharge'][current_state_index] > strategy['charge'][current_state_index]) & (strategy['discharge'][current_state_index] > strategy['idle'][current_state_index]):
        action_list.append("discharge")
    if (strategy['idle'][current_state_index] > strategy['charge'][current_state_index]) & (strategy['idle'][current_state_index] > strategy['discharge'][current_state_index]):
        action_list.append("idle")

print(action_list)
states_data['action'] = "not defined"
for i in range(len(action_list)):
    states_data['action'][i] = action_list[i]

states_data['original_cost'] = 0
for i in range(len(action_list)):
    if states_data['con_states'][i ] >= states_data['pv_states'][i ]:
        states_data['original_cost'][i] = states_data['price_state'][i] * (states_data['con_states'][i] - states_data['pv_states'][i])
    else:
        states_data['original_cost'][i] = 0

states_data['original_electricity_consumption'] = 0
for i in range(len(action_list)):
    states_data['original_electricity_consumption'][i] = max(states_data['con_states'][i] - states_data['pv_states'][i], 0)

# states_data['original_cost'] = 0
# states_data['original_electricity_consumption'] = 0
# battery_level = 0
# for i in range(len(action_list)):
#     if battery_level >= (states_data['con_states'][i ] - states_data['pv_states'][i ]):
#         states_data['original_cost'][i ] = 0
#         states_data['original_electricity_consumption'][i ] = 0
#         battery_level -= (states_data['con_states'][i ] - states_data['pv_states'][i ])
#     else:
#         states_data['original_cost'][i ] = states_data['price_state'][i ] * (states_data['con_states'][i ] - states_data['pv_states'][i ] - battery_level)
#         states_data['original_electricity_consumption'][i ] = max(states_data['con_states'][i ] - states_data['pv_states'][i ] - battery_level, 0)
#         battery_level = 0

states_data['new_electricity_consumption'] = 0
states_data['new_cost'] = 0
battery_level = 0
for i in range(len(action_list)):
    if states_data['action'][i ] == "charge":
        if states_data['con_states'][i ] >= states_data['pv_states'][i ]:
            states_data['new_cost'][i ] = states_data['price_state'][i ] * (states_data['con_states'][i ] - states_data['pv_states'][i ] + 3)
            states_data['new_electricity_consumption'][i ] = states_data['con_states'][i ] + 3
            battery_level += 3
        else:
            states_data['new_cost'][i ] = states_data['price_state'][i ] * 3
            states_data['new_electricity_consumption'][i ] = 3
            battery_level += states_data['pv_states'][i  ] - states_data['con_states'][i ] + 3
    if states_data['action'][i ] == "discharge":
        if battery_level >= (states_data['con_states'][i ] - states_data['pv_states'][i ]):
            states_data['new_cost'][i ] = 0
            states_data['new_electricity_consumption'][i ] = 0
            battery_level -= (states_data['con_states'][i ] - states_data['pv_states'][i ])
        else:
            states_data['new_cost'][i ] = states_data['price_state'][i ] * (states_data['con_states'][i ] - states_data['pv_states'][i ] - battery_level)
            states_data['new_electricity_consumption'][i ] = max(states_data['con_states'][i ] - states_data['pv_states'][i ] - battery_level, 0)
    if states_data['action'][i ] == "idle":
        if states_data['con_states'][i ] >= states_data['pv_states'][i ]:
            states_data['new_cost'][i ] = states_data['price_state'][i ] * (states_data['con_states'][i ] - states_data['pv_states'][i ])
            states_data['new_electricity_consumption'][i ] = max(states_data['con_states'][i ] - states_data['pv_states'][i ], 0)
            battery_level += 0
        else:
            states_data['new_cost'][i ] = 0
            states_data['new_electricity_consumption'][i ] = 0
            battery_level += states_data['pv_states'][i ] - states_data['con_states'][i ]

states_data.to_csv("./optimization_data.csv", index=False)
