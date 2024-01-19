import pandas as pd

input_csv_file = '../dataset/opsd-household_data-2020-04-15/household_data_60min_singleindex.csv'
df = pd.read_csv(input_csv_file)

residential_1 = ['utc_timestamp', 'DE_KN_residential1_dishwasher', 'DE_KN_residential1_freezer', 'DE_KN_residential1_grid_import', 'DE_KN_residential1_heat_pump', 'DE_KN_residential1_pv', 'DE_KN_residential1_washing_machine']

selected_data = df[residential_1]

output_csv_file = './data_generated/residential_power/power_of_residential_1.csv'
selected_data.to_csv(output_csv_file, index=False)

print("first generation")

residential_2 = ['utc_timestamp','DE_KN_residential2_circulation_pump', 'DE_KN_residential2_dishwasher', 'DE_KN_residential2_freezer', 'DE_KN_residential2_grid_import', 'DE_KN_residential2_washing_machine']

selected_data = df[residential_2]

output_csv_file = './data_generated/residential_power/power_of_residential_2.csv'
selected_data.to_csv(output_csv_file, index=False)

print("second generation")

residential_3 = ['utc_timestamp', 'DE_KN_residential3_circulation_pump', 'DE_KN_residential3_dishwasher', 'DE_KN_residential3_freezer', 'DE_KN_residential3_grid_export',
                 'DE_KN_residential3_grid_import', 'DE_KN_residential3_pv', 'DE_KN_residential3_refrigerator', 'DE_KN_residential3_washing_machine']

selected_data = df[residential_3]

output_csv_file = './data_generated/residential_power/power_of_residential_3.csv'
selected_data.to_csv(output_csv_file, index=False)

print("third generation")

residential_4 = ['utc_timestamp', 'DE_KN_residential4_dishwasher', 'DE_KN_residential4_ev', 'DE_KN_residential4_freezer', 'DE_KN_residential4_grid_export','DE_KN_residential4_grid_import',
                 'DE_KN_residential4_pv', 'DE_KN_residential4_heat_pump', 'DE_KN_residential4_pv', 'DE_KN_residential4_refrigerator', 'DE_KN_residential4_washing_machine']

selected_data = df[residential_4]

output_csv_file = './data_generated/residential_power/power_of_residential_4.csv'
selected_data.to_csv(output_csv_file, index=False)

print("fourth generation")

residential_5 = ['utc_timestamp', 'DE_KN_residential5_dishwasher', 'DE_KN_residential5_grid_import', 'DE_KN_residential5_refrigerator', 'DE_KN_residential5_washing_machine']

selected_data = df[residential_5]

output_csv_file = './data_generated/residential_power/power_of_residential_5.csv'
selected_data.to_csv(output_csv_file, index=False)

print("fifth generation")

residential_6 = ['utc_timestamp', 'DE_KN_residential6_circulation_pump', 'DE_KN_residential6_dishwasher', 'DE_KN_residential6_freezer', 'DE_KN_residential6_grid_export', 'DE_KN_residential6_grid_import', 'DE_KN_residential6_pv', 'DE_KN_residential6_washing_machine']

selected_data = df[residential_6]

output_csv_file = './data_generated/residential_power/power_of_residential_6.csv'
selected_data.to_csv(output_csv_file, index=False)

print("sixth generation")

################### claer the empty data #####################

file_path = ['./data_generated/residential_power/power_of_residential_1.csv', './data_generated/residential_power/power_of_residential_2.csv', './data_generated/residential_power/power_of_residential_3.csv',
             './data_generated/residential_power/power_of_residential_4.csv', './data_generated/residential_power/power_of_residential_5.csv', './data_generated/residential_power/power_of_residential_6.csv']

for file in file_path:
    data = pd.read_csv(file)
    data = data.dropna(subset=data.columns[1:], how='all')
    data.to_csv(file, index=False)
    print(file + " is cleared")

################### calculate the changes #####################

for f in file_path:
    data = pd.read_csv(f)
    diff_data = pd.DataFrame()
    diff_data[data.columns[0]] = data[data.columns[0]]

    for column in data.columns[1:]:
        diff_data[column] = data[column].diff()

    diff_data.to_csv(f[:-4] + '_change.csv', index=False)
    print(f + " is calculated")



