import pandas as pd
from pathlib import Path

# Define the list of files you want to include in the merge
weather_files = [
    "data/weather/dunkerque/francefull.csv",
    "data/weather/easington/birminghamfull.csv",
    "data/weather/germany/germanyfull.csv",
    "data/weather/nybro/nybrofull.csv",
    "data/weather/zeebrugge/begiumfull.csv"
]

columns_to_keep = [
    "time",
    "temperature_2m (°C)",
    "wind_speed_10m (km/h)",
    "wind_direction_10m (°)",
    "cloud_cover (%)"
]

# Initialize an empty list to store dataframes
dfs = []

# Iterate over the files in the weather_files list
for file_path in weather_files:
    file_path = Path(file_path)
    location = file_path.parents[0]  # Assuming the location is in the immediate parent directory
    location_name = "_".join(location.parts[-1:])  # Using only the immediate parent directory as identifier

    try:
        # Read CSV, explicitly handling bad lines and skipping irrelevant header rows
        df = pd.read_csv(file_path, delimiter=",", on_bad_lines='skip', engine="python")

        # Ensure 'time' column exists and filter by columns to keep
        if "time" not in df.columns:
            print(f"Skipping {file_path} - No 'time' column found.")
            continue

        # Keep only the necessary columns
        df = df.loc[:, [col for col in columns_to_keep if col in df.columns]]

        # Rename columns with the location prefix, except for 'time'
        df = df.rename(columns={col: f"{col}_{location_name}" for col in df.columns if col != "time"})

        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Merge all dataframes on 'time'
if dfs:
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="time", how="outer")

    # Save to CSV for verification
    merged_df.to_csv("merged_weather_data_targeted.csv", index=False)
    print("Merged CSV saved as 'merged_weather_data_targeted.csv'.")
else:
    print("No valid CSV files found for merging.")