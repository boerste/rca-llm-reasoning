#!/bin/bash

# Datasets
DATASETS=("GAIA" "OpenRCA-Market-cloudbed-1" "OpenRCA-Market-cloudbed-2")

# Dataset specific preparation steps
python prep_GAIA_dataset.py \
    --config config.yaml

python prep_OpenRCA_Market_dataset.py \
    --config config.yaml \
    --dataset OpenRCA-Market-cloudbed-1

python prep_OpenRCA_Market_dataset.py \
    --config config.yaml \
    --dataset OpenRCA-Market-cloudbed-2

for DATASET in "${DATASETS[@]}"; do
    # Preprocess the data to extract pre-fault (normal) data and post-fault injection data 
    python extract_pre_and_post_fault_telemetry.py \
        --config config.yaml \
        --dataset $DATASET

    # Train detectors & Drain log miner
    python train_metric_trace_detectors.py \
        --config config.yaml \
        --dataset $DATASET --stage both
    python train_log_template_miner.py \
        --config config.yaml \
        --dataset $DATASET

    # Extract alerts
    python alert_extractor.py \
        --config config.yaml \
        --dataset $DATASET
done