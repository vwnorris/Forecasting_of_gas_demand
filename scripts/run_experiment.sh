#!/bin/bash

model_name="rnn"
seq_len=192
target_column="Volumrate_Easington_hourly,Volumrate_St_Fergus_hourly,Volumrate_Belgia_hourly,Volumrate_Frankrike_hourly,Volumrate_Tyskland_hourly"
data_path="../data/ready/weather.csv"
epochs=500
log_file="experiment_results_${model_name}.log"
echo "Starting experiments..." > "$log_file"

# Extended hyperparameters
learning_rates=(0.001)
batch_sizes=(16)
pred_lens=(24)

for pred_len in "${pred_lens[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "🔄 Running experiment with pred_len=$pred_len, learning_rate=$lr, batch_size=$batch_size, model=$model_name" | tee -a "$log_file"
            
            python3 -u ../run_normalized.py \
                --model "$model_name" \
                --dataset "$data_path" \
                --target_column "$target_column" \
                --target_columns "$target_column" \
                --seq_len "$seq_len" \
                --pred_len "$pred_len" \
                --learning_rate "$lr" \
                --batch_size "$batch_size" \
                --epochs "$epochs" \
                --device "cuda" | tee -a "$log_file"
        done
    done
done

echo "All experiments completed! Results saved in $log_file"