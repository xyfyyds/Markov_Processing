import pandas as pd
import matplotlib.pyplot as plt
# 加载CSV文件
df = pd.read_csv('./optimization_data.csv')

# 如果日期时间不仅包含日期，还包含时间，需要将其转换为日期
df['Date'] = pd.to_datetime(df['Time']).dt.date

# 按日期分组
grouped = df.groupby('Date')

all = 0
op = 0
saving_cost = 0
original_costs = []
Markov_costs = []
savings = 0.0
# 对每个日期的第五列和第六列分别计算总和，并比较
for date, group in grouped:
    sum_ori = group['original_cost'].sum()
    sum_Markov = group['new_cost'].sum()
    original_costs.append(sum_ori)
    Markov_costs.append(sum_Markov)
    saving_cost += sum_ori - sum_Markov
    if (sum_ori - sum_Markov) / abs(sum_ori) > 0.000001 or (sum_ori - sum_Markov) / abs(sum_ori) < -0.000001:
        savings += (sum_ori - sum_Markov) / abs(sum_ori)
    if sum_ori > sum_Markov:
        op += 1
    all+=1

print('The number of days: ', all)
print('The probability of a success optimization: ', op/all * 100, '%')
print('The average saving cost: ', saving_cost/all)
print('The average saving rate: ', savings/all * 100, '%')

plt.figure(figsize=(20, 12))

plt.plot(original_costs[:100], label='Original costs')
plt.plot(Markov_costs[:100], label='Markov costs')

plt.xlabel('Days')
plt.ylabel('Costs')
plt.title('Optimization results')

plt.legend()
plt.show()