import itertools

import pandas as pd

# Define the possible values each element in the array can take
possible_values = [-1, 0, 1]

decisions = []
dd = list(itertools.product(possible_values, repeat=12))
for item in dd:
    decisions.append(list(item))

file_path = '../data_generated/weather/temperature_price_states.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)
price = data['DK_1_price_day_ahead'][66:78].values
print(price)

file_path = '../data_generated/residential_power/per_pc_residential_1.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)
consumption = (data['total_consumption'][0:12] * 6).values
print(consumption)

file_path = '../data_generated/residential_power/pv_residential_1.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)
s = data['DE_KN_residential1_pv'][8:20]
solar_power = s.values
print(solar_power)

best_cost = 1000000000
best_decision = []
amount = 3.0
cap = [0]
power_from_grid = []

for decision in decisions:
    cost = 0.0
    for i in range(12):
        if decision[i] == 1:
            if i != 0:
                cap.append(cap[i - 1] + amount + solar_power[i])
            else:
                cap.append(amount + solar_power[i])
            power_direct = consumption[i]
            cost += price[i] * (power_direct + amount)
        if decision[i] == 0:
            if i != 0:
                cap.append(cap[i - 1] + solar_power[i])
            else:
                cap.append(solar_power[i])
            power_direct = consumption[i]
            cost += price[i] * power_direct
        if decision[i] == -1:
            if cap[i] == 0:
                decision[i] = 0
            if i != 0:
                cap.append(cap[i-1] + solar_power[i] - min(amount, consumption[i], cap[i-1]))
                power_direct = consumption[i] - min(amount, consumption[i], cap[i-1])
                cost += price[i] * power_direct
            else:
                cap.append(solar_power[i] - min(amount, consumption[i], 0))
                power_direct = consumption[i]
                cost += price[i] * power_direct
    # print(cost)
    # print(decision)
    if cost < best_cost:
        best_cost = cost
        best_decision.clear()
        best_decision.append(decision)
    if cost == best_cost:
        best_decision.append(decision)

print(best_decision)
print(best_cost)
