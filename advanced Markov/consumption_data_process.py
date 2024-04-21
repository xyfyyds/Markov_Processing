import pandas as pd
import numpy as np

df = pd.read_csv('../data_generated/residential_power/per_pc_residential_1.csv')

df['consumption'] = df.iloc[:, 1] * 6


df['con_states'] = np.round(df['consumption'] / 0.25) * 0.25
df['con_states'] = df['con_states'].replace(0, 0.25)

df['Time'] = pd.to_datetime(df['utc_timestamp'])

data = df[['Time', 'Hour', 'consumption', 'con_states']]

data.to_csv('../data_generated/residential_power/consumption_data.csv', index=False)