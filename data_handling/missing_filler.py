import pandas as pd

# Load the merged dataset
data = pd.read_csv("merged_econ_with_flow.csv", parse_dates=["date"])

# Check for NaN values
nan_counts = data.isna().sum()
print("Missing values per column:\n", nan_counts[nan_counts > 0])

data = data.interpolate(method="linear", limit_direction="both")  # Linear interpolation for continuous time-series
data.fillna(method="ffill", inplace=True)  # Forward fill as backup for categorigal/discrete data
data.fillna(0, inplace=True)  # Fill remaining NaNs with 0s

# Save the cleaned dataset
fileName = "cleaned_econ_multi.csv"
data.to_csv(f"{fileName}", index=False)

print(f"Missing values handled. Cleaned dataset saved as '{fileName}'.")
