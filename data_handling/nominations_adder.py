import pandas as pd

### Function to add nominations to dataset if they are missing ###

gassco = pd.read_csv("data/FromGassco.csv", parse_dates=["Oslo Time"])
gassco.set_index("Oslo Time", inplace=True)
gassco_hourly = gassco.resample("h").mean().reset_index()

nomination_columns = [
    "Nominasjoner Easington [MSM3]",
    "Nominasjoner St. Fergus [MSM3]",
    "Nominasjoner Danmark [MSM3]",
    "Nominasjoner Belgia [MSM3]",
    "Nominasjoner Frankrike [MSM3]",
    "Nominasjoner Tyskland [MSM3]"
]
columns_to_keep = ["Oslo Time"] + nomination_columns
gassco_hourly = gassco_hourly[columns_to_keep]

# Example dataset for adding nominations
complete = pd.read_csv("data/ready/econ_daytype.csv", parse_dates=["date"])

merged = complete.merge(gassco_hourly, left_on="date", right_on="Oslo Time", how="left")
merged.drop(columns=["Oslo Time"], inplace=True)

for col in nomination_columns:
    if col in merged.columns:
        merged[col] = merged[col].fillna(method="ffill").fillna(0)

merged.to_csv("data/ready/econ_daytype_noms.csv", index=False)
print("🍞 Merging and NaN handling complete. Saved as 'complete_with_nominations.csv'.")
