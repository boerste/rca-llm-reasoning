"""
Train Metric and Trace Anomaly Detectors

Purpose:
- Prepare normal (non-fault) data for metrics and traces from either a pre-fault window
    or an explicitly marked non-anomalous period
- Train metric detectors (mean/std) and trace detectors (Isolation Forest per flow/feature)
- Persist trained detectors for downstream alert extraction

Prerequisites:
- processed-data/pre-fault-data exists (built by extract_pre_and_post_fault_telemetry.py),
    or normal-period is set in config.yaml to define a baseline window; optionally
    processed-data/normal-data may already exist
- gt-file points to the label.csv produced by the dataset prep script
    (prep_GAIA_dataset.py, prep_OpenRCA_Market_dataset.py);
    if run-table is present, labels are indexed by orig_fault_id
- processed-data directory exists and is writable; detector artifacts are saved under processed-data/detector

Workflow Stages:
1) Prepare: aggregate normal metrics per (entity, kpi) and traces per (source, destination, operation)
2) Train: compute per-KPI (mean, std) and fit Isolation Forests for duration/5xx/4xx counts

Inputs/Config:
- YAML config (--config) with dataset paths; 
- dataset key selected by --dataset
- Uses normal period if available (in config); otherwise derives normal data from pre-fault windows

Outputs:
- normal-metrics.pkl, normal-traces.pkl under processed-data/detector/
- metric-detector.json and trace-detector/ directory containing fitted models

Examples:
    python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset GAIA --stage both
    python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset OpenRCA-Market-cloudbed-1 --stage both
    python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset OpenRCA-Market-cloudbed-2 --stage both
"""
import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from utils.log_util import get_logger

from utils import io_util
from extractor.trace_alert_extractor import slide_window

logger = get_logger(__name__)
TRACE_WINDOW_SIZE = 30 * 1000 # 30 seconds in milliseconds

def prepare_normal_data_from_pre_fault(pre_data: dict, label_df: pd.DataFrame) -> tuple:
    """
    Extract normal (pre-fault) metric and trace data from pre-fault datasets.

    Aggregates per-fault slices into consolidated structures used for training
    detectors. Metrics are grouped by (entity, metric) and concatenated across
    faults; traces are grouped by (parent_name, service_name, operation) with
    operation normalized to exclude query parameters.

    Args:
        pre_data (dict): Mapping fault_id -> { 'metric': pd.DataFrame, 'trace': pd.DataFrame }.
        label_df (pd.DataFrame): Fault injection/label table including `fault_id`.

    Returns:
        tuple of (dict, dict):
            - normal_metrics: Dict[entity][kpi] -> pd.DataFrame (concatenated across faults)
            - normal_traces: Dict[source-destination-operation] -> list of pd.DataFrame
    """
    normal_metrics = defaultdict(lambda: defaultdict(list))
    normal_traces = defaultdict(list)

    for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Preparing training data"):
        chunk = pre_data[row['fault_id']]
        
        # Metrics
        metric_df = chunk['metric']
        if not metric_df.empty:
            metric_gp = metric_df.groupby(['entity', 'metric'])
            for (entity, kpi), group_df in metric_gp:
                normal_metrics[entity][kpi].append(group_df)

        # Traces
        trace_df = chunk['trace']
        if 'operation' in trace_df.columns:
            trace_df['operation'] = trace_df['operation'].str.split('?').str[0]
            trace_gp = trace_df.groupby(['parent_name', 'service_name', 'operation'])
            for (source, destination, operation), group_df in trace_gp:
                key = f"{source}-{destination}-{operation}"
                normal_traces[key].append(group_df)

    # Concatenate metric dataframes for each entity/kpi
    for entity in normal_metrics:
        for kpi in normal_metrics[entity]:
            normal_metrics[entity][kpi] = pd.concat(normal_metrics[entity][kpi], ignore_index=True)

    return dict(normal_metrics), dict(normal_traces)

def prepare_normal_data_from_non_anomalous(normal_data: dict) -> tuple:
    """
    Extract normal metric and trace data from a non-anomalous period dataset.

    Args:
        normal_data (dict): Mapping with keys 'metric' and 'trace' holding DataFrames.

    Returns:
        tuple of (dict, dict):
            - normal_metrics: Dict[entity][kpi] -> pd.DataFrame
            - normal_traces: Dict[source-destination-operation] -> list of pd.DataFrame
    """
    normal_metrics = defaultdict(dict)
    normal_traces = defaultdict(list)
    
    # Metrics
    metric_df = normal_data['metric']
    metric_gp = metric_df.groupby(['entity', 'metric'])
    for (entity, kpi), group_df in metric_gp:
        normal_metrics[entity][kpi] = group_df

    # Traces
    trace_df = normal_data['trace']
    if 'operation' in trace_df.columns:
        trace_df['operation'] = trace_df['operation'].str.split('?').str[0]
        trace_gp = trace_df.groupby(['parent_name', 'service_name', 'operation'])
        for (source, destination, operation), group_df in trace_gp:
            key = f"{source}-{destination}-{operation}"
            normal_traces[key].append(group_df)
                
    return dict(normal_metrics), dict(normal_traces)
    
def train_metric_detectors(normal_metrics: dict) -> dict:
    """
    Train mean/std detectors for each `(entity, kpi)`.

    Computes per-KPI mean and standard deviation across the normal period.

    Args:
        normal_metrics (dict): Dict[entity][kpi] -> pd.DataFrame with value column.

    Returns:
        dict: Dict[entity][kpi] -> [mean, std] used by k-sigma alerting.
    """
    detectors = {}
    for entity, kpi_dict in normal_metrics.items():
        detectors[entity] = {}
        for kpi, df in kpi_dict.items():
            detectors[entity][kpi] = [
                df['value'].mean(), 
                df['value'].std()
            ]
    return detectors


def train_trace_detectors(
    normal_traces: dict,
    win_size=TRACE_WINDOW_SIZE,
    stride=TRACE_WINDOW_SIZE // 2, # Integer division, ensure true multiple
    min_samples=10,
    min_std=1e-3,
    n_estimators=1000
) -> dict:
    """
    Train Isolation Forest detectors for trace flows with sufficient variance and size.

    Builds features via sliding windows per flow: mean duration, 500 count, 400 count.
    Skips duration detector when variance is below `min_std`. Each detector is
    trained on a single feature vector shaped `(n_samples, 1)`.

    Args:
        normal_traces (dict): Dict[source-destination-operation] -> list of pd.DataFrame.
        win_size (int): Sliding window size in milliseconds.
        stride (int): Stride in milliseconds.
        min_samples (int): Minimum windows required to train for a flow.
        min_std (float): Minimum std required to train the duration detector.
        n_estimators (int): Number of trees for each IsolationForest.

    Returns:
        dict: Dict[flow_name] -> {'dur_detector', '500_detector', '400_detector'}.
    """
    trace_detectors = {}

    for flow_name, call_dfs in tqdm(normal_traces.items(), desc="Training trace detectors"):
        mean_durations, error_500_counts, error_400_counts = [], [], []
        for df in call_dfs:
            _, durs, err_500, err_400 = slide_window(df, win_size, stride)
            mean_durations.extend(durs)
            error_500_counts.extend(err_500)
            error_400_counts.extend(err_400)

        enough_samples = (len(mean_durations) >= min_samples)
        if not enough_samples:
            logger.warning(f"Skipped all for {flow_name} due to insufficient samples ({len(mean_durations)}).")
            continue
        
        duration_std = np.std(mean_durations)
        enough_variation_duration = (duration_std >= min_std)
        
        error_500_std = np.std(error_500_counts)
        error_400_std = np.std(error_400_counts)
        
        detectors = {}
        if enough_variation_duration:
            detectors['dur_detector'] = IsolationForest(n_estimators=n_estimators, n_jobs=-1, random_state=0).fit(np.array(mean_durations).reshape(-1, 1))
        else:
            logger.warning(f"Skipped dur_detector for {flow_name} due to insufficient variance (std = {duration_std}).")

        detectors['500_detector'] = IsolationForest(n_estimators=n_estimators, n_jobs=-1, random_state=0).fit(np.array(error_500_counts).reshape(-1, 1))
        detectors['400_detector'] = IsolationForest(n_estimators=n_estimators, n_jobs=-1, random_state=0).fit(np.array(error_400_counts).reshape(-1, 1))

        if detectors:
            trace_detectors[flow_name] = detectors

    return trace_detectors


def main():
    """
    CLI entry point to prepare normal data and train detectors.
    - Parses config and dataset options, 
    - prepares normal metrics/traces (from non-anomalous or pre-fault data),
    - trains mean/std metric detectors and Isolation Forest trace detectors,
    - and saves artifacts to the detector dir.
    """
    parser = argparse.ArgumentParser(description="Train metric and trace detectors from pre-failure data.")
    parser.add_argument("--config", type=str, default="code/alert-extractor/config.yaml", help="Path to config YAML.")
    parser.add_argument("--dataset", type=str, default="OpenRCA-Market-cloudbed-1", help="Dataset name in config.")
    parser.add_argument("--stage", type=str, default="both", choices=["prepare", "train", "both"], help="Which stage to run.")
    args = parser.parse_args()

    config = io_util.load_config(args.config)
    dataset_config = config[args.dataset]
    detector_dir = os.path.join(dataset_config['processed-data'], 'detector')
    os.makedirs(detector_dir, exist_ok=True)
    
    has_non_anomalous_period = dataset_config.get('normal-period', False)

    logger.info(f"Starting metric and trace detector training for {args.dataset}")
    logger.info("Loading data...")
    
    # If non-anomalous data exists, use it to prepare normal data. Else, use the pre-fault data.
    if has_non_anomalous_period:
        logger.info("Non-anomalous period exists")
        logger.info('Loading data from normal-data')
        data = io_util.load_normal_data(
            os.path.join(dataset_config['processed-data'], 'normal-data'),
            modalities_to_skip=['log'])
    else:
        logger.info('Loading data from pre-fault-data')
        data = io_util.load_fault_data(
            os.path.join(dataset_config['processed-data'], 'pre-fault-data'),
            modalities_to_skip=['log'])
    
    # Use the pre-filtered fault injection table (if available) to accurately extract "normal" non-fault data
    if dataset_config.get('run-table', False):
        logger.info(f"Loading {dataset_config['run-table']} for labels.")
        label_df = pd.read_csv(dataset_config['run-table'])
    else:
        logger.info(f"Loading {dataset_config['gt-file']} for labels.")
        label_df = pd.read_csv(dataset_config['gt-file'])

    # Step 1: Prepare normal data
    if args.stage in ["prepare", "both"]:
        logger.info("Preparing normal data...")
        if has_non_anomalous_period:
            normal_metrics, normal_traces = prepare_normal_data_from_non_anomalous(data)
        else:
            normal_metrics, normal_traces = prepare_normal_data_from_pre_fault(data, label_df)
        logger.info(f"Saving normal metric and trace data to normal-metrics.pkl and normal-traces.pkl in {detector_dir}")
        io_util.save(os.path.join(detector_dir, "normal-metrics.pkl"), normal_metrics)
        io_util.save(os.path.join(detector_dir, "normal-traces.pkl"), normal_traces)

    # Step 2: Train detectors
    if args.stage in ["train", "both"]:
        logger.info(f"Loading normal metric and trace data from {detector_dir}...")
        normal_metrics = io_util.load(os.path.join(detector_dir, "normal-metrics.pkl"))
        normal_traces = io_util.load(os.path.join(detector_dir, "normal-traces.pkl"))

        logger.info("Training metric detectors...")
        metric_detectors = train_metric_detectors(normal_metrics)
        io_util.save_json(os.path.join(detector_dir, "metric-detector.json"), metric_detectors)
        logger.info(f"Metric detector training complete: detector saved to {detector_dir}")
        
        logger.info("Training trace detectors...")
        trace_detectors = train_trace_detectors(normal_traces)
        io_util.save_trace_detectors(os.path.join(detector_dir, "trace-detector"), trace_detectors)
        logger.info(f"Trace detector training complete: detector saved to {detector_dir}")


    logger.info(f"Finished detector training. Results saved to {detector_dir}")

# python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset GAIA --stage both
# python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-1" --stage both
# python code/alert-extractor/train_metric_trace_detectors.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-2" --stage both
if __name__ == "__main__":
    main()
