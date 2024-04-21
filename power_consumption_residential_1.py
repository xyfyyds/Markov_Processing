import pandas as pd

file_path = './data_generated/residential_power/power_of_residential_1_changes.csv'
data = pd.read_csv(file_path)

data['total_consumption'] = data.iloc[:, [1, 2, 4, 6]].sum(axis=1)

selected_columns = ['utc_timestamp', 'DE_KN_residential1_dishwasher', 'DE_KN_residential1_freezer', 'DE_KN_residential1_heat_pump', 'DE_KN_residential1_washing_machine', 'total_consumption']

data[selected_columns].to_csv('./data_generated/residential_power/power_consumption_residential_1.csv', index=False)

file_path = './data_generated/residential_power/power_consumption_residential_1.csv'
data = pd.read_csv(file_path)

data.fillna(0, inplace=True)

data.to_csv('./data_generated/residential_power/power_consumption_residential_1.csv', index=False)
