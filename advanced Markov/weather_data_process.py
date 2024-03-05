import csv
from datetime import datetime
import pandas as pd

# # 读取txt文件
# with open(
#         '../../weather datasets/stundenwerte_TU_00078_20041101_20221231_hist/produkt_tu_stunde_20041101_20221231_00078.txt', 'r') as txt_file:
#     lines = txt_file.readlines()
#
# # 提取表头和数据
# header = lines[0].strip().split(';')
# data_lines = [line.strip().split(';') for line in lines[1:]]
#
# for row in data_lines:
#     # 第二列是时间，假设是CSV文件中的第1个数据列（索引为0）
#     date_int = int(row[1])
#     # 转化为日期时间对象
#     date_obj = datetime.strptime(str(date_int), '%Y%m%d%H')
#     # 用转化后的对象替换原始整数
#     row[1] = date_obj
#
# # 写入CSV文件
# with open('../data_generated/weather/temperature.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(header)
#     csv_writer.writerows(data_lines)
#
#
# # 读取txt文件
# with open(
#         '../../weather datasets/stundenwerte_ST_01684_row/produkt_st_stunde_20010101_20240131_01684.txt', 'r') as txt_file:
#     lines = txt_file.readlines()
#
# # 提取表头和数据
# header = lines[0].strip().split(';')
# data_lines = [line.strip().split(';') for line in lines[1:]]
#
# for row in data_lines:
#     date_str = row[1]
#     date_obj = datetime.strptime(date_str, '%Y%m%d%H:%M')
#     row[1] = date_obj
#
# # 写入CSV文件
# with open('../data_generated/weather/solar.csv', 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(header)
#     csv_writer.writerows(data_lines)


# 读取文件A和文件B，假设时间列是第二列
df_a = pd.read_csv('../data_generated/weather/temperature.csv', parse_dates=[1])
df_b = pd.read_csv('../../dataset/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv', parse_dates=[0])

df_a['Time'] = pd.to_datetime(df_a['MESS_DATUM'])
print(df_a['Time'])
df_b['Time_'] = pd.to_datetime(df_b['utc_timestamp']).dt.tz_localize(None)
print(df_b['Time_'])

df_a['temperature'] = df_a['TT_TU']

# 根据时间列合并两个DataFrame，根据A文件的"x"列和B文件的"y"列进行匹配
merged_df = pd.merge(df_a, df_b, left_on=['Time'], right_on=['Time_'], how='inner')

cleaned_df = merged_df[['Time', 'DK_1_price_day_ahead', 'temperature']].dropna(subset=['DK_1_price_day_ahead'])

# 将合并结果写入新的CSV文件
cleaned_df.to_csv('../data_generated/weather/temperature_price.csv', index=False)

