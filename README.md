# Time Series Forecasting of European Gas Demand
This repository contains the code and data used for testing out different datasets and deep learning models for forecasting gas flow rates at key European Terminals. The pipeline includes preprocessing of data, training and evaluating models, normalization practices and visualizations of results. 
<br>

**Note**: Target variables have been excluded from the repository due to data sharing restrictions.
> This work was carried out as part of a Master’s thesis the Norwegian University of Science and Technology, focusing on time series forecasting of energy demand for Gassco AS.


---

## 📂 Folder Structure

### `data/`
Contains all data used to make the final datasets, and the datasets actually used in experiments. 

### `data/ready/`
Contains preprocessed CSV datasets ready for model training and evaluation:
- `complete_noms.csv`, `econ_noms.csv`, `weather.csv`: Datasets with different feature groups.
- `*_noms.csv`: Making sure that economic and complete data contains nomination features. 
- `*_daytype.csv`: Datasets with additional engineered features like day types - did not yield better results..

### `models/`
Houses the core model definitions:
- `ModernTCN.py`, `PETformer.py`, `SegRNN.py`, `KANAD.py`(Unused), `CNN.py`, `RNN.py`, etc.
- `weights/`: Directory for saved model checkpoints.
- `weights/best/`: Contains a .json file for each of the three main datasets, containing best performance for each model for each horizon.

### `scripts/`
Contains shell script that runs the entire pipeline - and .log files containing the log of the last run of each model. These show the training and evaluation in action. 

### `results/`
Contains the Granger Causality results for each terminal, in different formats. 

### `predictions/` and `predictions_consolidated/`
Contain raw model forecasts and consolidated results per target. The consolidated one contains shorter term plots.

### `logs/`
Includes training logs for all used models with and without normalization, as well as KAN-AD with normalized data and PETformer before Denmark was removed as a target variable. 

---

## 📀 Merging the datasets

### 1. Merge weather data
All weather CSV files from different stations are merged into a single dataframe using `data_merger.py`. Columns are renamed with location-specific prefixes, and merging is done on the time column using an outer join. The result is saved as `merged_weather_data.csv`.

### 2. Combine x with volume rates
The flow data (`FromGassco.csv`) is resampled to hourly frequency using a mean aggregation and merged with data X (either weather data or price data) on timestamp using `flow_to_weather.py`. This produces `merged_weather_with_flow.csv`.

### 3. Fill NaN values
The final step in data cleaning is handled by `missing_filler.py`. Missing values in the merged dataset are imputed using a 3-phase strategy:

1. Linear interpolation for continuous time-series features.

2. Forward fill for discrete or faulty values.

3. Final fallback to zero for any remaining missing entries.

---

## 🚀 Running the Project

### 1. Run Experiments via Shell Script

To run all defined experiments using a shell script:

```bash
cd scripts
./run_projects.sh
```
### 2. The Normalized pipelines
The run_projects.sh file contains variables that can be changed for different experiments. 

```python
model_name="segrnn" #Select the model for training - Alternatives are "cnn", "rnn", "segrnn", "moderntcn" and "petformer"
seq_len=192 #192 hours was used for all final experiments
target_column="Volumrate_Easington_hourly,Volumrate_St_Fergus_hourly,Volumrate_Belgia_hourly,Volumrate_Frankrike_hourly,Volumrate_Tyskland_hourly" #Chosen target columns, separated by comma
data_path="../data/ready/weather.csv" #All datasets used are in this folder. Change "weather.csv" if wanted to "econ_noms.csv" or "complete_noms.csv"
epochs=500 #Maximum epochs, early stopping is available
...
# Extended hyperparameters
learning_rates=(0.01 0.001 0.0001)
batch_sizes=(16 32)
pred_lens=(24 48 72)
```
It automatically:

- Loads the normalized dataset

- Applies data splits

- Trains the specified model

- Evaluates performance on 24h, 48h, and 72h prediction horizons

- Saves results to the results/ folder

### 3. The non-normalized pipeline of `run.py`
The run file `run.py` was early in the process found to be inferior to the normalized running, and here abandoned. It is therefore some changes behind `run_normalized.py`.

---
## 🧠 Available Models

- **CNN**  
  A baseline Temporal Convolutional Network (TCN)-style model for time series regression. 

- **RNN**  
  A standard Recurrent Neural Network for sequence modeling.

- **ModernTCN**  
  A Temporal Convolutional Network with RevIN normalization and patch embedding for time-series inputs.

- **PETformer**  
  A Transformer-based model designed to handle periodic patterns in long-horizon forecasting.

- **SegRNN**  
  An RNN-based encoder-decoder that predicts future flow rates using sequence embeddings.

> The model `KANAD.py` was included in the repository but not used in final experiments.

--- 

## 📊 Results
- Performance metrics are saved in results/ as .json files (e.g. best_parameters.json, best_easington.json).

- Forecast outputs are located in predictions/ and predictions_consolidated/.

- Visualizations such as time series plots and feature importance are saved in the plots/ directory.

---

## 🛠️ Requirements
- torch >= 1.10.0
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- scikit-learn>=0.24.0

Dependencies can be installed with:
```bash
pip install -r requirements.txt
```
--- 
## 🛡️License

This repository contains code and data developed as part of a Master's thesis at the Norwegian University of Science and Technology (NTNU). The materials are intended for academic and research purposes only.

Unauthorized commercial use, distribution, or modification of the contents is prohibited without prior written consent from the author.

For inquiries regarding usage rights or collaborations, please contact vic@norris.no or louislinnerud@gmail.com.
