import pandas as pd

### Adds the datetime timefeatures to selected dataset ###

# Load the data
data_path = 'data/ready/weather.csv'
daytype_path = 'data/economic/timefeatures_1h.csv'
data = pd.read_csv(data_path)
daytype = pd.read_csv(daytype_path)

# Function to correct the year in the date string
def correct_year(date_str):
    parts = date_str.split('-')
    if len(parts[0]) == 3: 
        parts[0] = '2' + parts[0] 
    return '-'.join(parts)

# Correct and parse the corrected date column
data['date'] = data['date'].apply(correct_year)
daytype['date'] = daytype['date'].apply(correct_year)

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
daytype['date'] = pd.to_datetime(daytype['date'], format='%Y-%m-%d %H:%M:%S')

# Merge the data
merged_data = pd.merge(data, daytype, on='date', how='left')

# Save the merged data
save_path = 'data/ready/weather_daytype.csv'
merged_data.to_csv(save_path, index=False)

print("Data merged and saved successfully.")
