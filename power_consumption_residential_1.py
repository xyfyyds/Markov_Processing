import pandas as pd

# 读取 CSV 文件
file_path = './data_generated/residential_power/power_of_residential_1_changes.csv'
data = pd.read_csv(file_path)

# 将第2、3、5、7列数据加在一起，生成新的一列
data['total_consumption'] = data.iloc[:, [1, 2, 4, 6]].sum(axis=1)

# 选择需要的列
selected_columns = ['utc_timestamp', 'DE_KN_residential1_dishwasher', 'DE_KN_residential1_freezer', 'DE_KN_residential1_heat_pump', 'DE_KN_residential1_washing_machine', 'total_consumption']

# 保存结果到新的 CSV 文件
data[selected_columns].to_csv('./data_generated/residential_power/power_consumption_residential_1.csv', index=False)

file_path = './data_generated/residential_power/power_consumption_residential_1.csv'
data = pd.read_csv(file_path)

# 将空白位置补0
data.fillna(0, inplace=True)

# 保存结果到新的 CSV 文件
data.to_csv('./data_generated/residential_power/power_consumption_residential_1.csv', index=False)
