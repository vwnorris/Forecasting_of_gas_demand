import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.KANAD import KANAD
from models.ModernTCN import ModernTCN as ModernTCN
from models.SegRNN import Model as SegRNN
from models.PETformer import PETformer
from models.RNN import RNN
from models.CNN import CNN
import math
import json
import csv
import time

# Function to load and normalize the dataset
def load_data(csv_file, target_columns, seq_len, pred_len, train_ratio=0.9, val_ratio=0.05):
    df = pd.read_csv(csv_file)

    # Remove unused columns if present
    columns_to_drop = [col for col in ["date", "Volumrate_Danmark_hourly"] if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Ensure all target columns exist
    for target in target_columns:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset!")
        
    # Add lagged versions of targets for prediction alignment
    for target in target_columns:
        df[f"{target}_lagged"] = df[target].shift(pred_len)

    # Drop rows with NaNs introduced by shifting
    df = df.dropna().reset_index(drop=True)

    # Separate features and targets
    feature_columns = [col for col in df.columns if col not in target_columns]
    feature_data = df[feature_columns].values
    target_data = df[target_columns].values

    # Generate sliding window sequences
    sequences, targets = [], []
    for i in range(len(feature_data) - seq_len - pred_len):
        sequences.append(feature_data[i:i+seq_len])
        targets.append(target_data[i+seq_len:i+seq_len+pred_len])

    sequences, targets = np.array(sequences), np.array(targets)

    # Split into train/val/test sets
    total_size = len(sequences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_X, train_y = sequences[:train_size], targets[:train_size]
    val_X, val_y = sequences[train_size:train_size+val_size], targets[train_size:train_size+val_size]
    test_X, test_y = sequences[train_size+val_size:], targets[train_size+val_size:]

    # Normalize features and targets using training data only
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    num_features = train_X.shape[2]
    train_X_reshaped = train_X.reshape(-1, num_features) 
    feature_scaler.fit(train_X_reshaped)

    train_y_reshaped = train_y.reshape(-1, len(target_columns))
    target_scaler.fit(train_y_reshaped)

    # Apply scaling
    def scale_X(X):
        batch_size, seq_len, num_features = X.shape
        X = feature_scaler.transform(X.reshape(-1, num_features)).reshape(batch_size, seq_len, num_features)
        return X

    def scale_y(y):
        batch_size, pred_len, num_targets = y.shape
        y = target_scaler.transform(y.reshape(-1, num_targets)).reshape(batch_size, pred_len, num_targets)
        return y

    train_X = scale_X(train_X)
    val_X = scale_X(val_X)
    test_X = scale_X(test_X)

    train_y = scale_y(train_y)
    val_y = scale_y(val_y)
    test_y = scale_y(test_y)

    return (train_X, train_y), (val_X, val_y), (test_X, test_y), df, feature_scaler, target_scaler


def plot_full_target_variable(target_data, train_size, val_size, target_name="", seq_len=96, pred_len=48):
    # Plot a full time series of the target variable split into training, validation, and testing segments
    
    # Define font sizes for various plot elements
    title_size = 18
    label_size = 18
    legend_size = 18

    # Set figure size
    plt.figure(figsize=(14, 6))

    # Calculate time index offset to align with prediction schedule
    offset = seq_len + pred_len
    time_index = pd.date_range(start="2020-01-01 06:00:00", periods=offset + len(target_data), freq="h")[offset:]
 
    # Sanity check to ensure length matches
    assert len(time_index) == len(target_data)

    # Define colors for different dataset segments
    colors = {
        "train": "#1f77b4",    # blue
        "val": "#4ca72c",      # green
        "test": "#d62728",     # red
        "split1": "#7f7f7f",   # grey
        "split2": "#5e5e0d"    # olive
    }

    # Plot each segment separately with corresponding colors
    plt.plot(time_index[:train_size], target_data[:train_size], label="Training Data", color=colors["train"])
    plt.plot(time_index[train_size:train_size + val_size], target_data[train_size:train_size + val_size], label="Validation Data", color=colors["val"])
    plt.plot(time_index[train_size + val_size:], target_data[train_size + val_size:], label="Testing Data", color=colors["test"])

    # Add vertical lines to mark train/val and val/test splits
    plt.axvline(x=time_index[train_size], color=colors["split1"], linestyle="--", label="Train-Validation Split")
    plt.axvline(x=time_index[train_size + val_size], color=colors["split2"], linestyle="--", label="Validation-Test Split")

    # Set plot labels and title
    plt.xlabel("Date", fontsize=label_size)
    plt.ylabel("Target Variable", fontsize=label_size)
    # plt.title(f"{target_name} Time Series with Train/Val/Test = 90%/5%/5%", fontsize=title_size)

    # Remove x-axis ticks and label for cleaner appearance
    ax = plt.gca()
    ax.set_xticks([])               
    ax.set_xticklabels([])          
    ax.set_xlabel("Time", fontsize=label_size)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("Volume rate [m³/h]", fontsize=label_size)

    # Add legend and format layout
    plt.legend(fontsize=legend_size)
    plt.tight_layout()
    plt.show()

experiment_results = []

# Main function handling training, evaluation and saving
def main():
    parser = argparse.ArgumentParser(description="Train different models on time-series data")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--target_column", type=str, required=True, help="Target variable column name")
    parser.add_argument("--target_columns", type=str, required=True, help="Comma-separated target variable column names")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on cpu")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of training data")
    
    args = parser.parse_args()

    target_columns = args.target_columns.split(",")

    # Load and preprocess data
    (train_X, train_y), (val_X, val_y), (test_X, test_y), df, feature_scaler, target_scaler = load_data(
    args.dataset, target_columns, args.seq_len, args.pred_len
    )

    # Convert to tensors
    train_X, train_y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    val_X, val_y = torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)
    test_X, test_y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_X, val_y), batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_y), batch_size=args.batch_size, shuffle=False)

    ########## Displays the full target variable(s) with train-validation-test split - commented for now ##########
    for target_idx, target_name in enumerate(target_columns):
        full_series = np.concatenate((
            train_y[:, :, target_idx].flatten(),
            val_y[:, :, target_idx].flatten(),
            test_y[:, :, target_idx].flatten()
        ))

        plot_full_target_variable(
            full_series,
            train_y.shape[0] * train_y.shape[1],
            val_y.shape[0] * val_y.shape[1],
            target_name=target_name,
            seq_len=args.seq_len,
            pred_len=args.pred_len
        )
    
    # Model config
    class Configs():
        seq_len = args.seq_len
        pred_len = args.pred_len
        enc_in = train_X.shape[2]  # Input features
        output_dim = train_y.shape[2]  # Number of target variables
        d_model = 64
        dropout = 0.25
        seg_len = 12
        task_name = "short_term_forecast"

    configs = Configs()

    # Initialize model
    if args.model.lower() == "segrnn":
        model = SegRNN(configs)
    elif args.model.lower() == "moderntcn":
        model = ModernTCN(configs=configs)
    elif args.model.lower() == "petformer":
        model = PETformer(configs)
    elif args.model.lower() == "rnn":
        model = RNN(configs)
    elif args.model.lower() == "cnn":
        model = CNN(configs)
    elif args.model.lower() == "kanad":
        model = KANAD(configs)
    else:
        raise ValueError("Unsupported model!")

    # Set loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move model to device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, switching to CPU.")
        args.device = "cpu"
    model.to(args.device)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            optimizer.zero_grad()
            output = model.forecast(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                val_output = model.forecast(x_val)
                val_loss = criterion(val_output, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0 
            torch.save(model.state_dict(), f"../models/weights/best_{args.model}_{args.pred_len}_normalized.pt")  # Save best model
        else:
            counter += 1 
            print(f"\033[93mNo improvement in validation loss for {counter}/{patience} epochs.\033[0m")

        if counter >= patience:
            print("\033[91mEarly stopping triggered! Training stopped.\033[0m")
            break

    completed_epochs = epoch + 1 - patience

    # Load best model
    model.load_state_dict(torch.load(f"../models/weights/best_{args.model}_{args.pred_len}_normalized.pt"))
    print("✅ Best model weights restored!")

    # Evaluation on test data
    print("✅ Training complete. Starting evaluation...")
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(args.device)
            output = model.forecast(x_batch).cpu().numpy()
            predictions.append(output)
            actuals.append(y_batch.cpu().numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Inverse transform
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, predictions.shape[2])).reshape(predictions.shape)
    actuals = target_scaler.inverse_transform(actuals.reshape(-1, actuals.shape[2])).reshape(actuals.shape)

    # Compute metrics per target variable
    num_targets = predictions.shape[-1]
    metrics_per_target = []

    for target_idx in range(num_targets):
        pred_trimmed = predictions[:, :, target_idx]
        act_trimmed = actuals[:, :, target_idx]

        # Flatten arrays
        pred_flat = pred_trimmed.flatten()
        act_flat = act_trimmed.flatten()

        # Prevent division by zero
        epsilon = 1e-8

        # Base metrics
        mae = np.mean(np.abs(pred_flat - act_flat))
        mse = np.mean((pred_flat - act_flat) ** 2)
        rmse = math.sqrt(mse)

        # Normalized RMSE
        mean_actual = np.mean(act_flat)
        nrmse = rmse / (mean_actual + epsilon)

        # R2 Score
        ss_res = np.sum((act_flat - pred_flat) ** 2)
        ss_tot = np.sum((act_flat - mean_actual) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))

        # MAPE with masking
        valid_mape_mask = np.abs(act_flat) > 1e-2
        if np.any(valid_mape_mask):
            mape = np.mean(np.abs(pred_flat[valid_mape_mask] - act_flat[valid_mape_mask]) / np.abs(act_flat[valid_mape_mask])) * 100
        else:
            mape = np.nan

        # SMAPE with masking
        denominator = np.abs(pred_flat) + np.abs(act_flat)
        valid_smape_mask = denominator > 1e-2
        if np.any(valid_smape_mask):
            smape = np.mean(2 * np.abs(pred_flat[valid_smape_mask] - act_flat[valid_smape_mask]) / denominator[valid_smape_mask]) * 100
        else:
            smape = np.nan

        # Append per-target results
        metrics_per_target.append({
            "Target Variable": target_columns[target_idx],
            "Epochs": args.epochs,
            "Prediction Length": args.pred_len,
            "Learning Rate": args.learning_rate,
            "Batch Size": args.batch_size,
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "nRMSE": nrmse,
            "R2": r2_score,
            "MAPE": mape if not np.isnan(mape) else "N/A",
            "SMAPE": smape if not np.isnan(smape) else "N/A",
            "Std.Dev": np.std(act_flat),
            "Mean": mean_actual
        })

    ########### Consolidate predicted results for all models in csv file ###########
    consolidated_dir = "../predictions_consolidated/"
    os.makedirs(consolidated_dir, exist_ok=True)

    for target_idx, metric in enumerate(metrics_per_target):
        station = metric["Target Variable"].replace("Volumrate_", "").replace("_hourly", "")
        mae = metric["MAE"]
        model_name = args.model.lower()
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace("complete_noms_", "")
        pred_len = args.pred_len

        filename = f"predictions_{pred_len}h_{station}_{dataset_name}.csv"
        save_path = os.path.join(consolidated_dir, filename)

        new_row = [f"{mae:.6f}"] + [f"{v:.4f}" for v in predictions[0, :, target_idx]]
        updated_rows = {}
        
        # Try loading existing data
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 0:
                        continue
                    model_key = row[0]
                    updated_rows[model_key] = row[1:]

        # Check if this model's result is better
        existing = updated_rows.get(model_name)
        if existing:
            existing_mae = float(existing[0])
            if mae < existing_mae:
                print(f"🔁 Overwriting {model_name} for {station} at {pred_len}h (MAE improved from {existing_mae:.4f} to {mae:.4f})")
                updated_rows[model_name] = [f"{mae:.6f}"] + [f"{v:.4f}" for v in predictions[0, :, target_idx]]
            else:
                continue
        else:
            print(f"➕ Adding {model_name} for {station} at {pred_len}h")
            updated_rows[model_name] = new_row

        # Write updated rows back to CSV
        with open(save_path, "w", newline='') as f:
            writer = csv.writer(f)
            for model_key, row in updated_rows.items():
                writer.writerow([model_key] + row)

    ####### Save actuals to CSV file ###########
    actuals_output_dir = "../actuals/"
    os.makedirs(actuals_output_dir, exist_ok=True)

    actuals_dict = {}  # station name -> actuals row

    for target_idx, target_name in enumerate(target_columns):
        station = target_name.replace("Volumrate_", "").replace("_hourly", "")
        actual_values = [f"{v:.4f}" for v in actuals[0, :, target_idx]]
        actuals_dict[station] = actual_values

    actuals_filename = f"actuals{args.pred_len}.csv"
    actuals_path = os.path.join(actuals_output_dir, actuals_filename)

    existing_actuals = {}
    if os.path.exists(actuals_path):
        with open(actuals_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    existing_actuals[row[0]] = row[1:]

    # Update if new or different
    for station, new_vals in actuals_dict.items():
        if station not in existing_actuals or existing_actuals[station] != new_vals:
            print(f"🛠️ Writing actuals for {station} in {actuals_filename}")
            existing_actuals[station] = new_vals

    # Write back all rows
    with open(actuals_path, "w", newline='') as f:
        writer = csv.writer(f)
        for station, row in existing_actuals.items():
            writer.writerow([station] + row)

    ########### Save predictions and actuals side-by-side ###########
    predictions_output_dir = "../predictions/"
    os.makedirs(predictions_output_dir, exist_ok=True)

    for target_idx, target_name in enumerate(target_columns):
        station = target_name.replace("Volumrate_", "").replace("_hourly", "")
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace("complete_noms_", "")
        pred_len = args.pred_len

        preds_flat = predictions[:, 0, target_idx]
        actuals_flat = actuals[:, 0, target_idx]

        df_pred_actual = pd.DataFrame({
            "Prediction": preds_flat,
            "Actual": actuals_flat
        })

        pred_csv_filename = (
            f"predictions_{pred_len}h_{station}_{dataset_name}_{args.model}_lr{args.learning_rate}_bs{args.batch_size}.csv"
        )
        pred_csv_path = os.path.join(predictions_output_dir, pred_csv_filename)

        df_pred_actual.to_csv(pred_csv_path, index=False)
        print(f"📤 Saved full predictions and actuals to {pred_csv_path}")
    ##### CSV saving is over ######

    # Convert results to DataFrame
    df_results = pd.DataFrame(metrics_per_target)

    # Print formatted results
    print("\n📊 Experiment Results Per Target Variable")
    print(df_results.to_string(index=False)) 

    # Paths for saving models
    weights_dir = "../models/weights/"
    best_weights_dir = f"../models/weights/best/{args.model}/"
    dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
    best_parameters_filename = f"best_parameters_{dataset_basename}_normalized.json"
    best_parameters_path = os.path.join("../models/weights/best/", best_parameters_filename)

    # Ensure directories exist
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(best_weights_dir, exist_ok=True)

    # Save model weights for every run
    model_save_path = os.path.join(weights_dir, f"{args.model}_{args.epochs}_normalized_weights.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model weights saved to {model_save_path}")
    best_models_list = []

    if os.path.exists(best_parameters_path) and os.path.getsize(best_parameters_path) > 0:
        with open(best_parameters_path, "r") as f:
            best_models_list = json.load(f)

    # Convert to dict: model_name -> entry
    best_models_dict = {entry["model"]: entry for entry in best_models_list}

    # Evaluate best per target variable
    for metric in metrics_per_target:
        target_name = metric["Target Variable"]
        mae = metric["MAE"]

        pred_len_key = f"targets{args.pred_len}"

        if args.model not in best_models_dict:
            best_models_dict[args.model] = {
                "model": args.model,
                pred_len_key: {}
            }

        model_entry = best_models_dict[args.model]

        if pred_len_key not in model_entry:
            model_entry[pred_len_key] = {}

        current_best = model_entry[pred_len_key].get(target_name)

        if current_best is None or mae < current_best["performance"]:
            if current_best is None:
                print(f"🎠 New best for {args.model} on {target_name} (MAE: {mae:.4f})")
            else:
                print(f"🎠 New best for {args.model} on {target_name} (MAE: {mae:.4f}, prev: {current_best['performance']:.4f})")

            model_entry[pred_len_key][target_name] = {
                "performance": float(mae),
                "R2": float(metric["R2"]),
                "MSE": float(metric["MSE"]),
                "RMSE": float(metric["RMSE"]),
                "SMAPE": float(metric["SMAPE"]) if metric["SMAPE"] != "N/A" else "N/A",
                "epochs": int(args.epochs),
                "completed_epochs": int(completed_epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "dataset": os.path.basename(args.dataset)
            }

            dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
            best_model_path = os.path.join(best_weights_dir, f"best_{args.model}_{target_name}_{dataset_basename}_{args.pred_len}_normalized.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"🥦 Saved new best weights to {best_model_path}")
        else:
            print(f"🌶️ No improvement for {args.model} on {target_name} (Current best MAE: {current_best['performance']:.4f})")

    best_models_list = list(best_models_dict.values())

    with open(best_parameters_path, "w") as f:
        json.dump(best_models_list, f, indent=4)

    print(f"📄 Updated best parameters json file")

    amount_of_targets = len(target_columns)
    results_filename = f"../logs/experiment_results_{args.model}_{amount_of_targets}_normalized.csv"
    df_results.to_csv(results_filename, mode='a', index=False, header=not os.path.exists(results_filename))
    return args
    
if __name__ == "__main__":
    start_time = time.time()
    args = main()   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ Total runtime: {elapsed_time/60:.2f} minutes")
