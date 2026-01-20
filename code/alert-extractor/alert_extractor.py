"""
Alert Extraction Orchestrator

Purpose:
- Load post-fault telemetry and trained detectors to produce metric, trace, and log alerts per fault.
- Support augmented labels by cloning alerts from original faults with timestamp shifts.

Prerequisites:
- Preprocessing and detector training is complete:
    - processed-data/post-fault-data exists (built by extract_pre_and_post_fault_telemetry.py)
    - Metric detector training: processed-data/detector/metric-detector.json exists (built by train_metric_trace_detectors.py)
    - Trace detector training: processed-data/detector/trace-detector/ directory exists (built by train_metric_trace_detectors.py)
    - Log template miner training: fault-alert-data/log/drain3-<system>-state.bin exists (built by train_log_template_miner.py)
- gt-file points to the labels CSV used here; if run-table is present, labels are indexed by orig_fault_id
- system, timezone, and optional thresholds (low-freq-p, max-logs-per-id, add-time-to-log-message) are set in config.yaml

Workflow:
1) Load labels and select non-augmented faults for primary alert generation.
2) Load post-fault telemetry for those fault ids.
3) Use metric and trace detectors and the Drain miner to extract alerts per fault.
4) If augmented faults exist, clone original alerts and shift timestamps accordingly.
5) Save alerts as JSON under the fault-alert-data directory.

Outputs:
- log/logs.json, metric/metrics.json, trace/traces.json written under the dataset's fault-alert-data directory.
"""
import time
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from utils import io_util
from utils.log_util import get_logger
from utils.time_util import convert_datetime_to_epoch, convert_epoch_to_datetime_str

from extractor.metric_alert_extractor import extract_metric_alerts
from extractor.trace_alert_extractor import extract_trace_alerts
from extractor.log_alert_extractor import extract_log_alerts
from drain.drain_template_extractor import init_drain

logger = get_logger(__name__)

def main():
    """
    CLI entry point to extract metric, trace, and log alerts per injected fault.

    Loads dataset configuration, labels, and post-fault telemetry; initializes or
    loads trained detectors and the Drain miner; generates alerts for each
    non-augmented fault; optionally clones alerts for augmented faults by
    shifting timestamps; and writes alert JSON files to the fault-alert-data directory.

    Args:
        None (arguments are parsed via argparse from the command line).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Alert Extraction")
    parser.add_argument('--config', default="code/extractor/config.yaml", type=str, help="Path to config file.")
    parser.add_argument('--dataset', default="OpenRCA-Market", type=str, help="Dataset to process.")
    args = parser.parse_args()

    logger.info("Starting alert extraction to produce final alerts.")
    start_time = time.time()

    # Load dataset configuration
    config = io_util.load_config(args.config)
    dataset = args.dataset
    dataset_config = config[dataset]

    dataset_dir = dataset_config['telemetry-data']
    processed_dir = dataset_config['processed-data']
    fault_alert_dir = dataset_config['fault-alert-data']
    gt_file = dataset_config['gt-file']

    logger.info("Loading fault labels")
    if dataset_config.get('run-table', False):
        logger.info(f"Loading {dataset_config['run-table']} for labels.")
        label_df = pd.read_csv(gt_file, index_col='orig_fault_id') # Index using the original fault id's for proper indexing
    else:
        logger.info(f"Loading {gt_file} for labels.")
        label_df = pd.read_csv(gt_file, index_col='fault_id')
        
    if 'augmented' in label_df.columns:
        label_df_no_augments = label_df[label_df['augmented'] != True]
    else:
        label_df_no_augments = label_df  # fallback to all rows if column is missing


    logger.info("Loading post-fault data")
    data = io_util.load_fault_data(
        os.path.join(processed_dir, 'post-fault-data'), 
        fault_ids_to_include=list(label_df.index)
    )
    logger.info(f"Data loaded, took {(time.time() - start_time) / 60} minutes")

    system = dataset_config['system']
    drain_persistence_file = os.path.join(fault_alert_dir, 'log', f"drain3-{system}-state.bin")

    logger.info("Loading detectors...")
    metric_detectors = io_util.load_json(os.path.join(processed_dir, 'detector', 'metric-detector.json'))
    trace_detectors = io_util.load_trace_detectors(os.path.join(processed_dir, 'detector', 'trace-detector'))
    miner = init_drain(system, drain_persistence_file)

    metric_alerts_dict = defaultdict(list)
    trace_alerts_dict = defaultdict(list)
    log_alerts_dict = defaultdict(list)
    metric_costs, trace_costs, log_costs = [], [], []

    for idx, row in tqdm(label_df_no_augments.iterrows(), total=label_df_no_augments.shape[0], desc="Processing data per injected fault (non-augmented)."):
        chunk = data[idx]
        # Determine fault_id from column if it exists, else fall back to index
        if 'fault_id' in label_df_no_augments.columns:
            fault_id = row['fault_id']
        else:
            fault_id = idx  # assume index holds fault_id

        # Extract metric alerts
        st = time.time()
        metric_df = chunk['metric']
        metric_alerts = []
        for entity_host, group_df in metric_df.groupby('entity'):
            kpi_alerts = extract_metric_alerts(entity_host,
                                               group_df,
                                               metric_detectors[entity_host],
                                               dataset)
            metric_alerts.extend(kpi_alerts)

        # Sort metric alerts by time, component, kpi
        metric_alerts.sort(key=lambda a: (a[0], a[1], a[3]))

        # Convert timestamp to string
        for alert in metric_alerts:
            alert[0] = str(alert[0])

        metric_alerts_dict[fault_id] = metric_alerts
        metric_costs.append(time.time() - st)

        # Extract trace alerts
        st = time.time()
        trace_df = chunk['trace']
        trace_alerts = extract_trace_alerts(trace_df,
                                            trace_detectors)
        trace_alerts_dict[fault_id] = trace_alerts
        trace_costs.append(time.time() - st)

        # Extract log alerts
        st = time.time()
        log_df = chunk['log']
        log_alerts = extract_log_alerts(log_df, 
                                        miner,
                                        dataset_config.get('low-freq-p', 0.5),
                                        dataset_config.get('max-logs-per-id', None),
                                        dataset_config.get('add-time-to-log-message', False),
                                        dataset_config['timezone'])
        log_alerts_dict[fault_id] = log_alerts
        log_costs.append(time.time() - st)
        
    if 'augmented' in label_df.columns:
        label_df = label_df.reset_index()
        label_df_augments = label_df[label_df['augmented'] == True]
        for idx, row in tqdm(label_df_augments.iterrows(), total=label_df_augments.shape[0], desc="Processing data per injected fault (augmented)."):
            fault_id = row['fault_id']
            orig_fault_id = row['orig_fault_id']

            # Find the matching fault_id with same orig_fault_id and not augmented
            matching_fault = label_df[
                (label_df['orig_fault_id'] == orig_fault_id) & 
                (label_df['augmented'] == False)
            ]

            if matching_fault.empty:
                logger.warning(f"Warning: No matching non-augmented fault found for fault_id={fault_id} with orig_fault_id={orig_fault_id}")
                continue

            fault_id_match = matching_fault.iloc[0]['fault_id']

            # Deep copy alerts from matching fault
            metric_alerts_dict[fault_id] = deepcopy(metric_alerts_dict.get(fault_id_match, []))
            trace_alerts_dict[fault_id]  = deepcopy(trace_alerts_dict.get(fault_id_match, []))
            log_alerts_dict[fault_id]    = deepcopy(log_alerts_dict.get(fault_id_match, []))

            orig_st_time =  convert_datetime_to_epoch(matching_fault.iloc[0]['fault_st_time'], dataset_config['timezone'])
            augmented_st_time = convert_datetime_to_epoch(row['fault_st_time'], dataset_config['timezone'])
            time_shift = augmented_st_time - orig_st_time

            def shift_time(alerts, shift_amount, alert_type):
                def shift_time_for_log_message(message):
                    if not message:
                        return message
                    try:
                        timestamp_str = message.split(" | ")[0]
                        timestamp_dt = convert_datetime_to_epoch(timestamp_str.replace(",", "."))
                        timestamp_shifted = timestamp_dt + shift_amount
                        timestamp_shifted_str = convert_epoch_to_datetime_str(timestamp_shifted).replace(".", ",")
                        return message.replace(timestamp_str, timestamp_shifted_str, 1)
                    except Exception as e:
                        logger.warning(f"Failed to shift log message: {message}, error: {e}")
                        return message
                
                def shift_time_for_alert(alert):
                    if alert_type in ['metric', 'trace']:
                        alert[0] = str(int(alert[0]) + shift_amount)
                    else: # log
                        messages: list[str] = alert[2]
                        repr_message: list[str] = alert[4]
                        alert[2] = [shift_time_for_log_message(message) for message in messages]
                        alert[4] = shift_time_for_log_message(repr_message)
                    return alert
                
                shifted_alerts = []
                for alert in alerts:
                    try:
                        shifted_alert = shift_time_for_alert(list(alert))
                        shifted_alerts.append(shifted_alert)
                    except Exception as e:
                        logger.warning(f"Failed to shift alert timestamp: {alert}, error: {e}")
                
                return shifted_alerts

            # Perform alert transformation (i.e., time shift)
            metric_alerts_dict[fault_id] = shift_time(metric_alerts_dict[fault_id], time_shift, 'metric')
            trace_alerts_dict[fault_id]  = shift_time(trace_alerts_dict[fault_id], time_shift, 'trace')
            log_alerts_dict[fault_id]    = shift_time(log_alerts_dict[fault_id], time_shift, 'log')
        
    logger.info(f'Cost to extract metric alerts: mean = {np.mean(metric_costs)}, total = {np.sum(metric_costs)}')
    logger.info(f'Cost to extract trace alerts: mean = {np.mean(trace_costs)}, total = {np.sum(trace_costs)}')
    logger.info(f'Cost to extract logs alerts: mean = {np.mean(log_costs)}, total = {np.sum(log_costs)}')


    io_util.save_json(os.path.join(fault_alert_dir, 'log/logs.json'), log_alerts_dict)
    io_util.save_json(os.path.join(fault_alert_dir, 'metric/metrics.json'), metric_alerts_dict)
    io_util.save_json(os.path.join(fault_alert_dir, 'trace/traces.json'), trace_alerts_dict)

    logger.info(f'Alert extraction completed!')
    logger.info(f'Total time taken: {(time.time() - start_time) / 60} minutes')

# python code/extractor/alert_extractor.py --config code/extractor/config.yaml --dataset GAIA
if __name__ == "__main__":
    main()