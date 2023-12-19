import pandas as pd

# 读取原始CSV文件
input_csv_file = 'C:\\Users\\XiangYF\\Desktop\\学习资料\\毕设\\source code\\dataset\\opsd-time_series-2020-10-06\\time_series_60min_singleindex.csv'
df = pd.read_csv(input_csv_file)

# 指定要提取的两列
selected_columns = ['utc_timestamp', 'cet_cest_timestamp', 'DE_LU_price_day_ahead']  # 将 'ColumnName2' 替换为实际的第二列名称

# 提取指定列的数据
selected_data = df[selected_columns]

# 保存结果到新的CSV文件
output_csv_file = './data_generated/price_of_DE_LU.csv'
selected_data.to_csv(output_csv_file, index=False)

# 读取原始CSV文件
input_csv_file = './data_generated/price_of_DE_LU.csv'
df = pd.read_csv(input_csv_file)

# 指定要检查的列（例如，'ColumnName' 替换为实际的列名）
column_to_check = 'DE_LU_price_day_ahead'

# 清除指定列中值为空的行
df_cleaned = df.dropna(subset=[column_to_check])

# 保存结果到新的CSV文件
output_csv_file = './data_generated/price_of_DE_LU_cleaned.csv'
df_cleaned.to_csv(output_csv_file, index=False)