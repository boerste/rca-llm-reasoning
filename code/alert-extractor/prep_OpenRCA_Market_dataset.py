"""
Prepare OpenRCA Market Datasets for Alert Extraction

Purpose:
- Normalize original labels into a common fault label schema
- Stage non-anomalous telemetry into a standard layout for detector training
    by copying or collapsing normal data from the provided normal telemetry source

Prerequisites:
- OpenRCA-Market dataset is already downloaded locally
- The alert-extractor config.yaml is updated so 'telemetry-data' and 'processed-data' point to local dataset directories
- For training non-anomalous baselines, normal-telemetry-data is set to a local path that contains the normal period data

Workflow:
1) Format original labels to fault_id, fault_level, fault_entity, fault_type,
     fault_st_time, fault_ed_time, fault_duration, fault_datetime
2) Copy normal logs and traces from the normal telemetry source into the dataset
3) Collapse normal metric directories into single CSVs per metric family

Outputs:
- label.csv with normalized labels
- telemetry/normal/log, telemetry/normal/trace, telemetry/normal/metric under the dataset directory
"""
import os
import shutil
import argparse
import pandas as pd
from utils.time_util import *
from utils import io_util
from utils.log_util import get_logger

logger = get_logger(__name__)

def format_labels_file(labels_file, save_file):
    """
    Normalize original labels into the common fault label schema.

    Sorts by timestamp and assigns a sequential fault_id, computes fault window
    fields, and renames columns to the standard fault_* names.

    Args:
        labels_file (str): Path to the dataset's original labels CSV.
        save_file (str): Output path for the normalized label CSV.
    """
    labels = pd.read_csv(labels_file)
    
    labels = labels.sort_values(by='timestamp').reset_index(drop=True)
    labels['fault_id'] = labels.index

    labels['fault_st_time'] = labels['timestamp'].astype(int)
    labels['fault_duration'] = 9 * 60  # 9 minutes in s
    labels['fault_ed_time'] = labels['fault_st_time'] + labels['fault_duration']

    labels = labels.rename(columns={"level": "fault_level", "component": "fault_entity", "reason": "fault_type", "datetime": "fault_datetime"})
    
    labels = labels[['fault_id', 'fault_level', 'fault_entity', 'fault_type', 'fault_st_time', 'fault_ed_time', 'fault_duration', 'fault_datetime']]
    labels.to_csv(save_file, index=False)
    
def format_telemetry_files(file_name_map, source_dir, save_dir, action):
    """
    Stage normal telemetry into the dataset directory via copy or directory collapse.

    When action is copy, copies specific source files to target filenames.
    When action is dir-collapse, concatenates all CSVs in a subdirectory into a
    single target file per mapping entry.

    Args:
        file_name_map (dict): Mapping of target filename to source filename or subdirectory.
        source_dir (str): Root directory of the source normal telemetry.
        save_dir (str): Destination directory to write staged files.
        action (str): One of copy or dir-collapse.
    """
    logger.info(f"Formatting files: '{action}' from {source_dir} to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    if action == 'copy':
        # Copy files from source_dir to save_dir according to the file_name_map
        for save_file, source_file in file_name_map.items():
            save_path = os.path.join(save_dir, save_file)
            source_path = os.path.join(source_dir, source_file)
            try:
                shutil.copy(source_path, save_path)
                logger.info(f"File '{source_path}' copied to '{save_path}' successfully.")
            except FileNotFoundError:
                logger.error(f"Error: Source file '{source_path}' not found.")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                
    if action == 'dir-collapse':
        # Collapse all files from source_dir into a single file and save in save_dir according to file_name_map
        for save_file, source_sub_dir in file_name_map.items():
            save_path = os.path.join(save_dir, save_file)
            source_sub_path = os.path.join(source_dir, source_sub_dir)
            
            df_list = []
            for source_file in os.listdir(source_sub_path):
                df = pd.read_csv(os.path.join(source_sub_path, source_file))
                df_list.append(df)
            
            df_all = pd.concat(df_list)
            df_all.to_csv(save_path, index=False)
                

def main():
    """
    CLI entry point to normalize labels and stage normal telemetry for OpenRCA Market.
    - Loads dataset config, 
    - writes normalized labels to fault-alerts, and
    - copies or collapses normal (non-anomalous) telemetry into the dataset directory under telemetry/normal.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', default="code/alert-extractor/config.yaml", type=str, help="Path to config file.")
    parser.add_argument('--dataset', default="OpenRCA-Market-cloudbed-2", type=str)
    args = parser.parse_args()
    dataset = args.dataset
    config = io_util.load_config(args.config)
    dataset_config = config[dataset]
    
    dataset_dir = dataset_config['telemetry-data']
    normal_telemetry_dir = dataset_config['normal-telemetry-data']
    
    # Normalize format of label file
    fault_alert_dir = dataset_config['fault-alert-data']
    original_labels_file = os.path.join(dataset_dir, dataset_config['original-labels'])
    label_file = os.path.join(fault_alert_dir, "label.csv")
    
    logger.info(f"Normalizing format of original label file: {original_labels_file}")
    format_labels_file(original_labels_file, label_file)
    logger.info(f"Saved formatted labels to {label_file}")

    # Pull non-anomalous data from normal_telemetry_dir into dataset_dir
    log_file_map = {
        'log_proxy.csv': 'log_filebeat-testbed-log-envoy.csv',
        'log_service.csv': 'log_filebeat-testbed-log-service.csv'
    }
    format_telemetry_files(
        log_file_map, 
        source_dir=os.path.join(normal_telemetry_dir, dataset_config['normal-logs-path']),
        save_dir=os.path.join(dataset_dir, 'telemetry', 'normal', 'log'),
        action='copy'
    )
    
    trace_file_map = {
        'trace_span.csv': 'trace_jaeger-span.csv'
    }
    format_telemetry_files(
        trace_file_map, 
        source_dir=os.path.join(normal_telemetry_dir, dataset_config['normal-traces-path']),
        save_dir=os.path.join(dataset_dir, 'telemetry', 'normal', 'trace'),
        action='copy'
    )

    metric_file_map = {
        'metic_container.csv': 'container',
        'metric_mesh.csv': 'istio',
        'metric_node.csv': 'node',
        'metric_runtime.csv': 'jvm',
        'metric_service.csv': 'service'
    }
    format_telemetry_files(
        metric_file_map, 
        source_dir=os.path.join(normal_telemetry_dir, dataset_config['normal-metrics-path']),
        save_dir=os.path.join(dataset_dir, 'telemetry', 'normal', 'metric'),
        action='dir-collapse'
    )
    
    logger.info("OpenRCA Market dataset prep complete.")
    
if __name__ == "__main__":
    main()


