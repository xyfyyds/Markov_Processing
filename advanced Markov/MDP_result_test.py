import pandas as pd
import matplotlib.pyplot as plt

'''
    testa nd plot the result
'''

df = pd.read_csv('./optimization_data.csv')

df['Date'] = pd.to_datetime(df['Time']).dt.date

grouped = df.groupby('Date')

all = 0
op = 0
saving_cost = 0
original_costs = []
Markov_costs = []
savings = 0.0

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

# Plot the first 100 lines of data, can be changed to plot the whole data
plt.plot(original_costs[:100], label='Original costs')
plt.plot(Markov_costs[:100], label='Markov costs')

plt.xlabel('Days')
plt.ylabel('Costs')
plt.title('Optimization results')

plt.legend()
plt.show()