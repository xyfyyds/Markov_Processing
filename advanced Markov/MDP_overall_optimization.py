import numpy as np


num_price_states = 20  # 电价状态数量
num_consumption_states = 15  # 消耗状态数量
num_solar_power_states = 20  # 太阳能状态数量
states = [(b, p, q) for b in range(num_price_states) for p in range(num_consumption_states) for q in range(num_solar_power_states)]  # 状态空间
actions = [0, 1, 2]  # 行动空间：0充电，1保持，2放电

# 随机生成电价状态转换矩阵
price_transition_matrix = np.random.rand(num_price_states, num_price_states)
price_transition_matrix = price_transition_matrix / price_transition_matrix.sum(axis=1)[:, None]  # 归一化

# 随机生成电价状态转换矩阵
consumption_transition_matrix = np.random.rand(num_consumption_states, num_consumption_states)
consumption_transition_matrix = consumption_transition_matrix / consumption_transition_matrix.sum(axis=1)[:, None]  # 归一化

# 随机生成电价状态转换矩阵
solar_power_transition_matrix = np.random.rand(num_solar_power_states, num_solar_power_states)
solar_power_transition_matrix = solar_power_transition_matrix / solar_power_transition_matrix.sum(axis=1)[:, None]  # 归一化

# 生成电价值，假设为0到1之间的均匀分布
prices = np.linspace(-10, 60, num_price_states)
consumption = np.linspace(0, 15, num_consumption_states)
solar_power = np.linspace(0, 5, num_solar_power_states)
battery_level = 0

# 初始化Q表
Q = np.zeros((len(states), len(actions)))


# 奖励函数，考虑新的状态和行动定义
def get_reward(battery_level, price_state, demand_state, solar_state, action):
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
    price_state = np.random.choice(num_price_states, p=price_transition_matrix[price_state])
    demand_state = np.random.choice(num_consumption_states, p=consumption_transition_matrix[demand_state])
    solar_state = np.random.choice(num_solar_power_states, p=solar_power_transition_matrix[solar_state])

    return battery_level, price_state, demand_state, solar_state, reward


# 更新Q-learning算法，考虑电价状态
episodes = 10000  # 更新训练回合数以适应更复杂的状态空间
epsilon = 0.1  # epsilon-贪婪策略
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子

for episode in range(episodes):
    # 随机初始化状态
    price_state, consumption_state, solar_power_state = np.random.choice(num_price_states), np.random.choice(num_consumption_states), np.random.choice(num_solar_power_states)
    state_index = states.index((price_state, consumption_state, solar_power_state))  # 状态索引
    battery_state = 0  # 电池状态
    done = False
    i = 0

    while i <= 100:
        # epsilon-贪婪策略
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q[state_index])

        # 奖励和下一个状态
        battery_state, price_state, consumption_state, solar_power_state, reward = get_reward(battery_state, price_state, consumption_state, solar_power_state, action)

        next_state_index = states.index((price_state, consumption_state, solar_power_state))
        # Q值更新
        Q[state_index, action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action])
        state_index = next_state_index

        i += 1


# 显示部分更新后的Q表
print(Q )
