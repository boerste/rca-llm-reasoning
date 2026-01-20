import os
import shutil
import argparse
import pandas as pd
from utils.time_util import *
from utils import io_util
from utils.log_util import get_logger

logger = get_logger(__name__)

def format_labels_file(labels_file, save_file):
    labels = pd.read_csv(labels_file)
    
    labels = labels.sort_values(by='timestamp').reset_index(drop=True)
    labels['fault_id'] = labels.index

    labels['fault_st_time'] = labels['timestamp'].astype(int) # unix timestamp in s
    labels['fault_duration'] = 5 * 60  # 5 minutes in s
    labels['fault_ed_time'] = labels['fault_st_time'] + labels['fault_duration']

    labels = labels.rename(columns={"level": "fault_level", "component": "fault_entity", "reason": "fault_type", "datetime": "fault_datetime"})
    
    labels = labels[['fault_id', 'fault_level', 'fault_entity', 'fault_type', 'fault_st_time', 'fault_ed_time', 'fault_duration', 'fault_datetime']]
    labels.to_csv(save_file, index=False)
                

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', default="code/extractor/config.yaml", type=str, help="Path to config file.")
    args = parser.parse_args()
    dataset = "OpenRCA-Telecom"
    config = io_util.load_config(args.config)
    dataset_config = config[dataset]
    
    dataset_dir = dataset_config['telemetry-data']

    # Normalize format of label file
    fault_alert_dir = dataset_config['fault-alert-data']
    os.makedirs(fault_alert_dir, exist_ok=True)
    original_labels_file = os.path.join(dataset_dir, dataset_config['original-labels'])
    label_file = os.path.join(fault_alert_dir, "label.csv")
    
    logger.info(f"Normalizing format of original label file: {original_labels_file}")
    format_labels_file(original_labels_file, label_file)
    logger.info(f"Saved formatted labels to {label_file}")

    
    logger.info("OpenRCA Telecom dataset prep complete.")
    
if __name__ == "__main__":
    main()


