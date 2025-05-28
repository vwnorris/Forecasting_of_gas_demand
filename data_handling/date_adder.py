import pandas as pd

### Function used to fill remaining dates in a CSV file when knowing the first date ###

model = "segrnn"
horizon = "24"
input_file = f"predictions/all_easington/{horizon}h_easington_{model}2.csv"
output_file = f"predictions/all_easington/{horizon}h_easington_dates_{model}2.csv"

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

start_date = pd.to_datetime(df['date'].dropna().iloc[0])
df['date'] = pd.date_range(start=start_date, periods=len(df), freq='h')

df.to_csv(output_file, index=False)
print(f"Updated CSV saved to: {output_file}")
