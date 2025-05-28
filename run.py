import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ModernTCN import ModernTCN as ModernTCN
from models.SegRNN import Model as SegRNN
from models.PETformer import PETformer
from models.RNN import RNN
from models.CNN import CNN
import math
import json

# Abandoned early for improved run_normalized version and therefore not commented throughout

def load_data(csv_file, target_columns, seq_len, pred_len, train_ratio=0.9, val_ratio=0.05):
    df = pd.read_csv(csv_file)
    columns_to_drop = [col for col in ["date", "Volumrate_Danmark_hourly"] if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    for target in target_columns:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset!")

    for target in target_columns:
        df[f"{target}_lagged"] = df[target].shift(pred_len)

    df = df.dropna().reset_index(drop=True)

    feature_columns = [col for col in df.columns if col not in target_columns]
    feature_data = df[feature_columns].values
    target_data = df[target_columns].values

    sequences, targets = [], []
    for i in range(len(feature_data) - seq_len - pred_len):
        sequences.append(feature_data[i:i+seq_len])
        targets.append(target_data[i+seq_len:i+seq_len+pred_len])

    sequences, targets = np.array(sequences), np.array(targets)

    total_size = len(sequences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  

    train_X, train_y = sequences[:train_size], targets[:train_size]
    val_X, val_y = sequences[train_size:train_size+val_size], targets[train_size:train_size+val_size]
    test_X, test_y = sequences[train_size+val_size:], targets[train_size+val_size:]

    return (train_X, train_y), (val_X, val_y), (test_X, test_y), df


def plot_full_target_variable(target_data, train_size, val_size, target_name=""):
    plt.figure(figsize=(12, 5))

    time_index = np.arange(len(target_data))

    plt.plot(time_index[:train_size], target_data[:train_size], label="Training Data", color="blue")

    plt.plot(time_index[train_size:train_size+val_size], target_data[train_size:train_size+val_size], label="Validation Data", color="green")

    plt.plot(time_index[train_size+val_size:], target_data[train_size+val_size:], label="Testing Data", color="red")

    plt.axvline(x=train_size, color="black", linestyle="--", label="Train-Validation Split")

    plt.axvline(x=train_size+val_size, color="grey", linestyle="--", label="Validation-Test Split")

    plt.xlabel("Time Steps")
    plt.ylabel("Target Variable")
    plt.title(f"{target_name} - Time Series with Train-Validation-Test Split")
    plt.legend()
    plt.show()

def plot_predictions_vs_actuals(predictions, actuals, seq_len, pred_len, model_name, batch_size, target_columns, comment, save_dir="../plots/"):
    save_dir = os.path.join(save_dir, model_name.lower())
    os.makedirs(save_dir, exist_ok=True) 

    num_targets = predictions.shape[-1] 
    
    for target_idx in range(num_targets):
        predictions_target = predictions[:, :, target_idx].flatten()
        actuals_target = actuals[:, :, target_idx].flatten()

        num_samples_short = min(pred_len, len(actuals_target), len(predictions_target))
        time_index_short = np.arange(num_samples_short)

        num_samples_full = min(len(actuals_target), len(predictions_target))
        time_index_full = np.arange(num_samples_full)

        plt.figure(figsize=(14, 6))
        plt.plot(time_index_short, actuals_target[:num_samples_short], label=f"Actual {target_columns[target_idx]}", color="blue", linewidth=1)
        plt.plot(time_index_short, predictions_target[:num_samples_short], label=f"Predicted {target_columns[target_idx]}", color="orange", linestyle="dashed", linewidth=1)
        plt.xlabel("Time Steps")
        plt.ylabel("Target Variable")
        plt.title(f"First {pred_len} Predictions vs. Actuals ({model_name}) - {target_columns[target_idx]}")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        save_path_short = os.path.join(save_dir, f"first_{pred_len}_results_{model_name.lower()}_{target_columns[target_idx]}_batch{batch_size}_seq{seq_len}_pred{pred_len}_time{timestamp}_{comment}.png")
        plt.savefig(save_path_short, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"📁 Short-term plot saved to {save_path_short}")

        plt.figure(figsize=(14, 6))
        plt.plot(time_index_full, actuals_target[:num_samples_full], label=f"Actual {target_columns[target_idx]}", color="blue", linewidth=1)
        plt.plot(time_index_full, predictions_target[:num_samples_full], label=f"Predicted {target_columns[target_idx]}", color="orange", linestyle="dashed", linewidth=1)
        plt.xlabel("Time Steps")
        plt.ylabel("Target Variable")
        plt.title(f"Full Test Set Predictions vs. Actuals ({model_name}) - {target_columns[target_idx]}")
        plt.legend()

        save_path_full = os.path.join(save_dir, f"full_test_results_{model_name.lower()}_{target_columns[target_idx]}_batch{batch_size}_seq{seq_len}_pred{pred_len}_time{timestamp}_{comment}.png")
        plt.savefig(save_path_full, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"📁 Full test set plot saved to {save_path_full}")

from sklearn.metrics import mean_squared_error

def permutation_feature_importance(model, val_loader, feature_columns, target_columns, device="cpu"):
    print("\n🔍 Computing Permutation Feature Importance...\n")

    val_X_batches, val_y_batches = [], []
    
    for _ in range(5):
        val_X, val_y = next(iter(val_loader))
        val_X_batches.append(val_X)
        val_y_batches.append(val_y)

    val_sample_X = torch.cat(val_X_batches, dim=0).to(device)
    val_sample_y = torch.cat(val_y_batches, dim=0).to(device)

    with torch.no_grad():
        baseline_preds = model.forecast(val_sample_X).cpu().numpy()

    feature_importances_per_target = {}

    for target_idx, target_name in enumerate(target_columns):
        baseline_mse = mean_squared_error(
            val_sample_y.cpu().numpy()[:, :, target_idx].flatten(),
            baseline_preds[:, :, target_idx].flatten()
        )

        feature_importances = []

        for feature_idx, feature_name in enumerate(feature_columns):
            val_sample_X_permuted = val_sample_X.clone()

            permuted_values = val_sample_X_permuted[:, :, feature_idx].cpu().numpy().flatten()
            np.random.shuffle(permuted_values)
            permuted_values = permuted_values.reshape(val_sample_X_permuted[:, :, feature_idx].shape)
            val_sample_X_permuted[:, :, feature_idx] = torch.tensor(permuted_values, dtype=torch.float32, device=device)

            with torch.no_grad():
                permuted_preds = model.forecast(val_sample_X_permuted).cpu().numpy()

            permuted_mse = mean_squared_error(
                val_sample_y.cpu().numpy()[:, :, target_idx].flatten(),
                permuted_preds[:, :, target_idx].flatten()
            )

            importance = permuted_mse - baseline_mse
            feature_importances.append((feature_name, importance))

        feature_importances.sort(key=lambda x: x[1], reverse=True)
        feature_importances_per_target[target_name] = feature_importances

    for target_name, feature_importances in feature_importances_per_target.items():
        print(f"\n🎯 Top 5 important features for target variable: {target_name}")
        for rank, (feature, importance) in enumerate(feature_importances[:5], start=1):
            print(f"{rank}. {feature} (Increase in MSE: {importance:.4f})")

experiment_results = []

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

    (train_X, train_y), (val_X, val_y), (test_X, test_y), df = load_data(
    args.dataset, target_columns, args.seq_len, args.pred_len
    )

    train_X, train_y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    val_X, val_y = torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)
    test_X, test_y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_X, val_y), batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_y), batch_size=args.batch_size, shuffle=False)

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
            target_name=target_name
        )
    
    class Configs():
        seq_len = args.seq_len
        pred_len = args.pred_len
        enc_in = train_X.shape[2]  
        output_dim = train_y.shape[2]  
        d_model = 64
        dropout = 0.25
        seg_len = 12
        task_name = "short_term_forecast"

    
    configs = Configs()

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
    else:
        raise ValueError("Unsupported model!")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, switching to CPU.")
        args.device = "cpu"
    
    model.to(args.device)

    best_val_loss = float('inf')
    patience = 3
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

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0 
            torch.save(model.state_dict(), f"../models/weights/best_{args.model}_{args.pred_len}.pt")  # Save best model
        else:
            counter += 1 
            print(f"\033[93mNo improvement in validation loss for {counter}/{patience} epochs.\033[0m")

        if counter >= patience:
            print("\033[91mEarly stopping triggered! Training stopped.\033[0m")
            break

    model.load_state_dict(torch.load(f"../models/weights/best_{args.model}_{args.pred_len}.pt"))
    print("✅ Best model weights restored!")

    feature_columns = [col for col in df.columns if col not in target_columns]

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
    num_targets = predictions.shape[-1]
    metrics_per_target = []

    for target_idx in range(num_targets):
        pred_trimmed = predictions[:, :, target_idx]
        act_trimmed = actuals[:, :, target_idx]

        pred_flat = pred_trimmed.flatten()
        act_flat = act_trimmed.flatten()

        epsilon = 1e-8

        mae = np.mean(np.abs(pred_flat - act_flat))
        mse = np.mean((pred_flat - act_flat) ** 2)
        rmse = math.sqrt(mse)

        mean_actual = np.mean(act_flat)
        nrmse = rmse / (mean_actual + epsilon)

        ss_res = np.sum((act_flat - pred_flat) ** 2)
        ss_tot = np.sum((act_flat - mean_actual) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))

        valid_mape_mask = np.abs(act_flat) > 1e-2
        if np.any(valid_mape_mask):
            mape = np.mean(np.abs(pred_flat[valid_mape_mask] - act_flat[valid_mape_mask]) / np.abs(act_flat[valid_mape_mask])) * 100
        else:
            mape = np.nan

        denominator = np.abs(pred_flat) + np.abs(act_flat)
        valid_smape_mask = denominator > 1e-2
        if np.any(valid_smape_mask):
            smape = np.mean(2 * np.abs(pred_flat[valid_smape_mask] - act_flat[valid_smape_mask]) / denominator[valid_smape_mask]) * 100
        else:
            smape = np.nan

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


    df_results = pd.DataFrame(metrics_per_target)

    print("\n📊 Experiment Results Per Target Variable")
    print(df_results.to_string(index=False))

    weights_dir = "../models/weights/"
    best_weights_dir = f"../models/weights/best/{args.model}/"
    dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
    best_parameters_filename = f"best_parameters_{dataset_basename}.json"
    best_parameters_path = os.path.join("../models/weights/best/", best_parameters_filename)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(best_weights_dir, exist_ok=True)

    model_save_path = os.path.join(weights_dir, f"{args.model}_{args.epochs}_weights.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model weights saved to {model_save_path}")

    best_models_list = []

    if os.path.exists(best_parameters_path) and os.path.getsize(best_parameters_path) > 0:
        with open(best_parameters_path, "r") as f:
            best_models_list = json.load(f)

    best_models_dict = {entry["model"]: entry for entry in best_models_list}

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
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "dataset": os.path.basename(args.dataset)
            }

            dataset_basename = os.path.splitext(os.path.basename(args.dataset))[0]
            best_model_path = os.path.join(best_weights_dir, f"best_{args.model}_{target_name}_{dataset_basename}_{args.pred_len}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"🥦 Saved new best weights to {best_model_path}")
        else:
            print(f"🌶️ No improvement for {args.model} on {target_name} (Current best MAE: {current_best['performance']:.4f})")

    best_models_list = list(best_models_dict.values())

    with open(best_parameters_path, "w") as f:
        json.dump(best_models_list, f, indent=4)

    print("📄 Updated best_parameters.json")


    amount_of_targets = len(target_columns)
    results_filename = f"../logs/experiment_results_{args.model}_{amount_of_targets}.csv"
    df_results.to_csv(results_filename, mode='a', index=False, header=not os.path.exists(results_filename))
    return args

    
if __name__ == "__main__":
    args = main()