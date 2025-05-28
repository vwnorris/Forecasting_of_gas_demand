import pandas as pd

### Merges economic, weather, and time features data into a single DataFrame ###

def load_and_prepare(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Load the data files
econ_data_path = 'data/ready/econ_multi.csv'
weather_data_path = 'data/weather/merged_weather_data.csv'
time_features_path = 'data/economic/timefeatures_1h.csv'

econ_data = load_and_prepare(econ_data_path)
weather_data = load_and_prepare(weather_data_path)
time_features = load_and_prepare(time_features_path)

# Merge the data on 'date', using outer join to ensure all data is included
merged_data = pd.merge(weather_data, econ_data, on='date', how='outer', suffixes=('_weather', '_econ'))
merged_with_day = pd.merge(merged_data, time_features, on='date', how='outer')

# Save the merged data without the index
save_path = 'data/ready/merged_data.csv'
merged_with_day.to_csv(save_path, index=False)

print("All data merged and saved successfully.")
