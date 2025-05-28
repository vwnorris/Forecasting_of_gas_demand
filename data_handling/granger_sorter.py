import pandas as pd

### Function to sort Granger top features by F-statistic ###

station = "Easington"
df = pd.read_csv(f"results/granger_top_features_complete_noms_{station}_with_f.csv")

# Sort by F-statistic descending
df_sorted = df.sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
df_sorted["Rank"] = range(1, len(df_sorted) + 1)

# Save to new CSV
df_sorted.to_csv(f"results/granger_top_features_{station}_fstat.csv", index=False)
print(f"🍣Printing complete for {station}")
