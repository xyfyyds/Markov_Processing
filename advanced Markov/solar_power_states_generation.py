import pandas as pd

df = pd.read_csv('../data_generated/weather/solar_generation.csv', parse_dates=[0])

for date in df['Time'].dt.date.unique():
    daily_data = df[df['Time'].dt.date == date]

    time_filtered_data = daily_data[(daily_data['Time'].dt.time >= pd.to_datetime('04:00:00').time()) &
                                    (daily_data['Time'].dt.time <= pd.to_datetime('18:00:00').time())]

    average = time_filtered_data.iloc[:, 5].mean()

    factor = 2.5 / average if average != 0 else 0

    df.loc[daily_data.index, df.columns[5]] = daily_data.iloc[:, 5] * factor

df['Hour'] = df['Time'].dt.hour

def round_to_nearest_five(value):
    return round(value / 5) * 5

df['solar_power_states'] = df.iloc[:, 3].apply(round_to_nearest_five)

def map_to_interval(value):
    if value <= 0.1:
        return 0
    interval_index = int((value - 0.0001) / 0.5)  #
    interval_mid = 0.25 + interval_index * 0.5
    return interval_mid

df['pv_states'] = df.iloc[:, 5].apply(map_to_interval)

df.to_csv('../data_generated/weather/solar_power_states.csv', index=False)
