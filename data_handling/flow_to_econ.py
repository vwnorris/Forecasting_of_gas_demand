import pandas as pd

### Function to load and merge flow data with economic data ###

flow_data = pd.read_csv("data/FromGassco.csv", parse_dates=["Oslo Time"])
flow_data.set_index("Oslo Time", inplace=True)

# Resample to hourly frequency
flow_data_hourly = flow_data.resample("h").mean()
flow_data_hourly.reset_index(inplace=True)

# Keep only the relevant column and rename them
flow_data_hourly = flow_data_hourly[["Oslo Time", 
                                     "Volumrate Easington [MSM3/D]", 
                                     "Volumrate St. Fergus [MSM3/D]",
                                     "Volumrate Danmark [MSM3/D]",
                                     "Volumrate Belgia [MSM3/D]",
                                     "Volumrate Frankrike [MSM3/D]",
                                     "Volumrate Tyskland [MSM3/D]",
                                     "Nominasjoner Easington [MSM3]",
                                     "Nominasjoner St. Fergus [MSM3]",
                                     "Nominasjoner Danmark [MSM3]",
                                     "Nominasjoner Belgia [MSM3]",
                                     "Nominasjoner Frankrike [MSM3]",
                                     "Nominasjoner Tyskland [MSM3]"]]
flow_data_hourly.rename(columns={"Volumrate Easington [MSM3/D]": "Volumrate_Easington_hourly"}, inplace=True)
flow_data_hourly.rename(columns={"Volumrate St. Fergus [MSM3/D]": "Volumrate_St_Fergus_hourly"}, inplace=True)
flow_data_hourly.rename(columns={"Volumrate Danmark [MSM3/D]": "Volumrate_Danmark_hourly"}, inplace=True)
flow_data_hourly.rename(columns={"Volumrate Belgia [MSM3/D]": "Volumrate_Belgia_hourly"}, inplace=True)
flow_data_hourly.rename(columns={"Volumrate Frankrike [MSM3/D]": "Volumrate_Frankrike_hourly"}, inplace=True)
flow_data_hourly.rename(columns={"Volumrate Tyskland [MSM3/D]": "Volumrate_Tyskland_hourly"}, inplace=True)

weather_data = pd.read_csv("data/economic/prices_hourly.csv", parse_dates=["Oslo Time"])
merged_data = weather_data.merge(flow_data_hourly, left_on="Oslo Time", right_on="Oslo Time", how="left")

# Save the final merged dataset
merged_data.to_csv("merged_econ_with_flow.csv", index=False)
print("Merged dataset saved as 'merged_econ_with_flow.csv'.")
