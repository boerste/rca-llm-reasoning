#!/bin/bash

# Datasets
DATASETS=("GAIA" "OpenRCA-Market-cloudbed-1" "OpenRCA-Market-cloudbed-2")

# Dataset specific preparation steps
python code/alert-extractor/prep_GAIA_dataset.py \
    --config code/alert-extractor/config.yaml

python code/alert-extractor/prep_OpenRCA_Market_dataset.py \
    --config code/alert-extractor/config.yaml \
    --dataset OpenRCA-Market-cloudbed-1

python code/alert-extractor/prep_OpenRCA_Market_dataset.py \
    --config code/alert-extractor/config.yaml \
    --dataset OpenRCA-Market-cloudbed-2

for DATASET in "${DATASETS[@]}"; do
    # Preprocess the data to extract pre-fault (normal) data and post-fault injection data 
    python code/alert-extractor/extract_pre_and_post_fault_telemetry.py \
        --config code/alert-extractor/config.yaml \
        --dataset $DATASET

    # Train detectors & Drain log miner
    python code/alert-extractor/train_metric_trace_detectors.py \
        --config code/alert-extractor/config.yaml \
        --dataset $DATASET --stage both
    python code/alert-extractor/train_log_template_miner.py \
        --config code/alert-extractor/config.yaml \
        --dataset $DATASET

    # Extract alerts
    python code/alert-extractor/alert_extractor.py \
        --config code/alert-extractor/config.yaml \
        --dataset $DATASET
done