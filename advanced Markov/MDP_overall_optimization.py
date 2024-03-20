import numpy as np
import pandas as pd

# 生成电价状态转换矩阵
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

print("prices states number: " + str(num_prices_states))
print("solar power states number: " + str(num_solar_power_states))
print("consumption states number: " + str(num_consumption_states))


battery_level = 0
states = [(b, p, q) for b in range(num_prices_states) for p in range(num_consumption_states) for q in range(num_solar_power_states)]  # 状态空间
actions = [0, 1, 2]  # 行动空间：0充电，1保持，2放电

# 初始化Q表
Q = np.zeros((len(states), len(actions)))

# possible solar power states in hour 4
file_path = '../data_generated/weather/solar_power_states.csv'
df = pd.read_csv(file_path)
unique_hour4_pv = df[df['Hour'] == 4]['pv_states'].unique()

# possible power consumption states in hour 0
file_path = '../data_generated/residential_power/consumption_data.csv'
df = pd.read_csv(file_path)
unique_hour0_con = df[df['Hour'] == 0]['con_states'].unique()

# 奖励函数，考虑新的状态和行动定义
def get_reward(battery_level, price_state, demand_state, solar_state, action, hour):
    price = prices[price_state]
    demand = consumption[demand_state]
    solar_production = solar_power[solar_state]

    # 计算奖励
    if action == 0:  # 充电
        reward = -price * (demand - solar_production) - price * 3
        battery_level += 3  # 限制电池电量在状态范围内
    elif action == 2:  # 放电
        reward = min(-price * (demand - solar_production - battery_level), 0)
        battery_level = max(0, battery_level - demand)  # 减少电池电量，但不低于0
    else:  # 保持
        reward = -price * (demand - solar_production)

    # 更新状态
    price_state = np.random.choice(num_prices_states, p=price_transition_matrix[price_state])
    demand_state = np.random.choice(num_consumption_states, p=consumption_transition_matrix_list[hour][demand_state])
    if 4 <= hour <= 17:
        solar_state = np.random.choice(num_solar_power_states, p=solar_power_transition_matrix_list[hour][solar_state])
    elif hour == 3:
        data = np.random.choice(unique_hour4_pv)
        solar_state = np.where(solar_power == data)[0][0]
    else:
        data = 0
        solar_state = np.where(solar_power == data)[0][0]
    hour = (hour + 1) % 24

    return battery_level, price_state, demand_state, solar_state, reward, hour


# Q-learning
episodes = 10000  # 训练回合数
epsilon = 0.1  # epsilon-贪婪策略
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子


for episode in range(episodes):
    # initialize the state, at time 0
    time = 0
    price_state, consumption_state, solar_power_state = np.random.choice(num_prices_states), np.where(consumption == np.random.choice(unique_hour0_con))[0][0], np.where(solar_power == 0)[0][0]
    state_index = states.index((price_state, consumption_state, solar_power_state))  # 状态索引
    battery_state = 0  # 电池状态
    done = False
    i = 0

    while i <= 1000:
        # epsilon-贪婪策略
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q[state_index])

        # 奖励和下一个状态
        battery_state, price_state, consumption_state, solar_power_state, reward, time = get_reward(battery_state, price_state, consumption_state, solar_power_state, action, time)

        next_state_index = states.index((price_state, consumption_state, solar_power_state))
        # Q值更新
        Q[state_index, action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action])
        state_index = next_state_index

        i += 1
        if battery_state >= 50:
            break


# 显示部分更新后的Q表
print(Q )

df = pd.DataFrame(Q, columns=['charge', 'idle', 'discharge'])
df.to_csv('strategy_by_Q_reward.csv', index=False)
