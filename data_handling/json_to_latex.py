import json
from collections import defaultdict

### Function to write the .json files into latex tables ###

input_json_path = "models/weights/best/best_parameters_weather_normalized.json"
output_latex_path = "models/weights/best/latex_prints/weather_results.tex"

model_name_map = {
    "cnn": "CNN",
    "rnn": "RNN",
    "segrnn": "SegRNN",
    "petformer": "PETformer",
    "moderntcn": "ModernTCN",
}

station_name_map = {
    "Volumrate_Easington_hourly": "Easington",
    "Volumrate_St_Fergus_hourly": "St Fergus",
    "Volumrate_Belgia_hourly": "Belgium",
    "Volumrate_Frankrike_hourly": "France",
    "Volumrate_Tyskland_hourly": "Germany",
}

with open(input_json_path, "r") as f:
    data = json.load(f)

dataset_name = data[0]["targets24"]["Volumrate_Easington_hourly"]["dataset"].split(".")[0].replace("_noms", "")

results = defaultdict(lambda: defaultdict(dict))
for model_entry in data:
    model_key = model_entry["model"]
    if model_key.lower() == "kanad":
        continue  # skip kanad
    model_name = model_name_map.get(model_key, model_key)
    for horizon_key in ["targets24", "targets48", "targets72"]:
        for station_key, metrics in model_entry[horizon_key].items():
            station_name = station_name_map.get(station_key, station_key)
            results[station_name][horizon_key][model_name] = metrics

latex = [
    f"\\section*{{Results for {dataset_name.capitalize()} Dataset}}"
]

for station, horizons in results.items():
    latex.append(f"\\subsection*{{Station: {station}}}")
    for horizon in ["targets24", "targets48", "targets72"]:
        latex.append(f"\\subsubsection*{{Prediction Horizon: {horizon[-2:]}}}")
        latex.append("\\begin{tabular}{lccccc}")
        latex.append("\\toprule")
        latex.append("Model & MAE & R$^2$ & MSE & RMSE & SMAPE \\\\")
        latex.append("\\midrule")
        for model in sorted(horizons[horizon]):
            metrics = horizons[horizon][model]
            row = f"{model} & " + " & ".join(
                f"{metrics[key]:.3f}" for key in ["performance", "R2", "MSE", "RMSE", "SMAPE"]
            ) + " \\\\"
            latex.append(row)
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\vspace{0.5cm}")

with open(output_latex_path, "w") as f:
    f.write("\n".join(latex))
# Emoji used for a splash of color in the terminal output.
print(f"🍇 LaTeX table written to {output_latex_path}")