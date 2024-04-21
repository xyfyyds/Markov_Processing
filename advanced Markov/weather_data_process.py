import csv
from datetime import datetime
import pandas as pd

'''
    process the weather data to a csv format
'''

with open(
        '../../weather datasets/stundenwerte_TU_00078_20041101_20221231_hist/produkt_tu_stunde_20041101_20221231_00078.txt', 'r') as txt_file:
    lines = txt_file.readlines()

header = lines[0].strip().split(';')
data_lines = [line.strip().split(';') for line in lines[1:]]

for row in data_lines:
    date_int = int(row[1])
    date_obj = datetime.strptime(str(date_int), '%Y%m%d%H')
    row[1] = date_obj

with open('../data_generated/weather/temperature.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(data_lines)

with open(
        '../../weather datasets/stundenwerte_ST_01684_row/produkt_st_stunde_20010101_20240131_01684.txt', 'r') as txt_file:
    lines = txt_file.readlines()

header = lines[0].strip().split(';')
data_lines = [line.strip().split(';') for line in lines[1:]]

for row in data_lines:
    date_str = row[1]
    date_obj = datetime.strptime(date_str, '%Y%m%d%H:%M')
    row[1] = date_obj

for row in data_lines:
    date_str = row[8]
    date_obj = datetime.strptime(date_str, '%Y%m%d%H:%M')
    row[8] = date_obj

with open('../data_generated/weather/solar.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(data_lines)


df_a = pd.read_csv('../data_generated/weather/temperature.csv', parse_dates=[1])
df_b = pd.read_csv('../../dataset/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv', parse_dates=[0])

df_a['Time'] = pd.to_datetime(df_a['MESS_DATUM'])
print(df_a['Time'])
df_b['Time_'] = pd.to_datetime(df_b['utc_timestamp']).dt.tz_localize(None)
print(df_b['Time_'])

df_a['temperature'] = df_a['TT_TU']

merged_df = pd.merge(df_a, df_b, left_on=['Time'], right_on=['Time_'], how='inner')

cleaned_df = merged_df[['Time', 'DK_1_price_day_ahead', 'temperature']].dropna(subset=['DK_1_price_day_ahead'])

cleaned_df.to_csv('../data_generated/weather/temperature_price.csv', index=False)


df_a = pd.read_csv('../data_generated/weather/solar.csv', parse_dates=[1])
df_b = pd.read_csv('../data_generated/residential_power/pv_residential_1.csv', parse_dates=[0])

df_a['Time'] = pd.to_datetime(df_a['MESS_DATUM_WOZ'])
df_a['Time'] = df_a['Time'] - pd.Timedelta(hours=1)
print(df_a['Time'])
df_b['Time_'] = pd.to_datetime(df_b['utc_timestamp']).dt.tz_localize(None)
print(df_b['Time_'])


merged_df = pd.merge(df_a, df_b, left_on=['Time'], right_on=['Time_'], how='inner')

cleaned_df = merged_df[['Time', 'ATMO_LBERG', 'FD_LBERG', 'FG_LBERG', 'SD_LBERG', 'DE_KN_residential1_pv']]

cleaned_df.to_csv('../data_generated/weather/solar_generation.csv', index=False)

