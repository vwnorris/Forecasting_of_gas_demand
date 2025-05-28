import pandas as pd
import os
from pathlib import Path

### Merges all available weather CSV files into a single weather dataframe ###

root_dir = "data/weather"
dfs = []  

# Walk through the directory structure
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = Path(subdir) / file
            location = Path(subdir).relative_to(root_dir)  
            location_name = "_".join(location.parts)

            try:
                # Read CSV, explicitly handling bad lines and skipping irrelevant header rows
                df = pd.read_csv(file_path, on_bad_lines='skip', engine="python")

                if "time" not in df.columns:
                    print(f"Skipping {file_path} - No 'time' column found.")
                    continue

                df = df.rename(columns={col: f"{col}_{location_name}" for col in df.columns if col != "time"})

                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if dfs:
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="time", how="outer")

    merged_df.to_csv("merged_weather_data.csv", index=False)
    print("Merged CSV saved as 'merged_weather_data.csv'.")

else:
    print("No valid CSV files found for merging.")
