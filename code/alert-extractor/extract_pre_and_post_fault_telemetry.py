"""
Extract Pre- and Post-Fault Telemetry Data

Purpose:
- Process raw traces, logs, and metrics from supported datasets (GAIA, OpenRCA-Market)
- Standardize fields and persist unified processed artifacts
- Slice pre- and post-fault telemetry windows per fault for downstream RCA and detector training

Prerequisites:
- Dataset has been downloaded locally and paths configured in the alert-extractor config.yaml
- Each dataset has been 'prepped' according to their respective 'prep' script
  (prep_GAIA_dataset.py, prep_OpenRCA_Market_dataset.py);
- Optional normal-period can be provided to export a baseline normal window

Workflow:
1) Build processed-data/trace.csv, processed-data/log.csv, processed-data/metric.parquet
    using telemetry-data with traces-path, logs-path, and metrics-path
2) Load labels from gt-file, or run-table when configured, and normalize time units
3) Compute pre/post windows per fault that avoid overlap with adjacent faults
4) Extract telemetry for each window and save pre-fault-data/, post-fault-data/, and optional normal-data/

Outputs:
- processed-data/trace.csv, processed-data/log.csv, processed-data/metric.parquet
- processed-data/pre-fault-data/ and processed-data/post-fault-data/ directories
- processed-data/normal-data/ when normal-period is defined

Examples:
    python code/extractor/extract_pre_and_post_fault_telemetry.py --config code/extractor/config.yaml --dataset GAIA
    python code/extractor/extract_pre_and_post_fault_telemetry.py --config code/extractor/config.yaml --dataset OpenRCA-Market-cloudbed-1
    python code/extractor/extract_pre_and_post_fault_telemetry.py --config code/extractor/config.yaml --dataset OpenRCA-Market-cloudbed-2
"""
import os
import pandas as pd
import warnings
import time
from tqdm import tqdm
from utils import io_util
from utils.time_util import convert_datetime_to_epoch
from utils.log_util import get_logger
import argparse
import glob
from typing import Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# Global window parameters (in milliseconds)
PRE_FAULT_WINDOW = 10 * 60 * 1000  # 10 minutes -- time window BEFORE injected fault start time
POST_FAULT_WINDOW = 10 * 60 * 1000  # 10 minutes -- time window AFTER injected fault start time

# Time buffers for after an injected fault ends
MIN_BUFFER = 200 # 200 ms

def process_traces(dir, save_dir, dataset, timezone):
    """
    Processes raw trace CSV files, adds parent span info, converts timestamps, and saves to trace.csv.
    
    Args:
        dir (str): Root directory containing raw trace CSV files.
        save_dir (str): Destination directory for processed artifacts.
        dataset (str): Dataset name, one of GAIA, OpenRCA-Market*.
        timezone (str): Timezone string used to convert GAIA timestamps to epoch ms.

    Returns:
        None
    """
    
    def get_trace_files(dir, dataset):
        if dataset == "GAIA":
            trace_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("2021-07.csv")]
            logger.info(f"Found {len(trace_files)} trace files in {dir} (only data from 2021-07)")
        if ("OpenRCA-Market" in dataset):
            trace_files = glob.glob(os.path.join(dir, "*"))
            logger.info(f"Found {len(trace_files)} trace files in {dir}")
        
        return trace_files
    
    def standardize_trace_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Standardize raw trace rows into a common schema.

        Args:
            df (pd.DataFrame): Input trace data for a single file.
            dataset (str): Dataset key to control field mapping.

        Returns:
            pd.DataFrame: Trace rows with required columns timestamp, trace_id, span_id,
            parent_span_id, start_time, end_time, duration, service_name, operation, status_code.
        """
        if dataset == "GAIA":
            df = df.rename(columns={
                'url': 'operation',
                'parent_id': 'parent_span_id'
            })
            df['start_time'] = df['start_time'].apply(lambda x: convert_datetime_to_epoch(str(x), timezone))
            df['end_time'] = df['end_time'].apply(lambda x: convert_datetime_to_epoch(str(x), timezone))
            df['duration'] = df['end_time'] - df['start_time']
            df['parent_span_id'] = df['parent_span_id'].fillna('-1').astype(str)  # Ensure not NaN and type str
            df = df[[
                'timestamp', 'trace_id', 'span_id', 'parent_span_id',
                'start_time', 'end_time', 'duration', 
                'service_name', 'operation', 'status_code'
            ]]
        elif "OpenRCA-Market" in dataset:
            df['start_time'] = df['timestamp']
            df['end_time'] = df['timestamp'] + df['duration']
            df = df.rename(columns={
                'cmdb_id': 'service_name',
                'parent_span': 'parent_span_id',
                'operation_name': 'operation',
            })
            df['parent_span_id'] = df['parent_span_id'].fillna('-1') # Ensure not NaN
            df['operation'] = df['operation'] + " (" + df['type'] + ")"
            
            # status_code: 0 (rpc, HTTP, telemetry), 200 (HTTP), Ok (db), OK (rpc), 4 (rpc), 13 (HTTP, rpc), 1 (rpc), 14 (rpc). 
            def map_status_code(val):
                """Map existing status codes to standardized 200/400/500"""
                if str(val) in ["0", "200", "OK", "Ok"]:
                    return 200
                elif str(val) in ["1", "4"]:
                    return 400
                elif str(val) in ["13", "14"]:
                    return 500
                else:
                    return 500  # default to 500 for unknown/error codes
            df['status_code'] = df['status_code'].apply(map_status_code)
            
            df = df[[
                'timestamp', 'trace_id', 'span_id', 'parent_span_id',
                'start_time', 'end_time', 'duration',
                'service_name', 'operation', 'status_code'
            ]]
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")
        
        return df

    def spans_df_left_join(trace_df: pd.DataFrame, dataset) -> pd.DataFrame:
        """
        Add parent service names via a self-join.

        Args:
            trace_df (pd.DataFrame): Standardized trace DataFrame.
            dataset (str): Dataset key for optional dataset-specific adjustments.

        Returns:
            pd.DataFrame: Trace rows with an extra parent_name column when resolvable.
        """
        spans_df_temp = trace_df.copy()
        parent_map = trace_df[['span_id', 'service_name']].rename(columns={'service_name': 'parent_name'})
        spans_df_temp = spans_df_temp.merge(
            parent_map, 
            left_on='parent_span_id', 
            right_on='span_id', 
            how='left'
        )
        spans_df_temp.drop(columns=['span_id_y'], inplace=True)
        spans_df_temp.rename(columns={'span_id_x': 'span_id'}, inplace=True)

        return spans_df_temp

    logger.info("Processing traces")
    save_file = os.path.join(save_dir, "trace.csv")
    
    trace_files = get_trace_files(dir, dataset)
    dfs = []
    for file_path in tqdm(trace_files, desc="Loading trace CSVs"):
        try:
            raw_df = pd.read_csv(file_path)
            df = standardize_trace_df(raw_df, dataset)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    trace_df = pd.concat(dfs)
    
    logger.info("Joining spans to add parent service info...")
    trace_df = spans_df_left_join(trace_df, dataset)
    
    # logger.info("Ordering by timestamp...")
    # trace_df = trace_df.sort_values(by="start_time").reset_index()
    logger.info("Saving results...")
    trace_df.to_csv(save_file, index=False)
    logger.info(f"Saved processed traces to {save_file}.")

def process_logs(dir, save_dir, dataset, timezone):
    """
    Build a unified log.csv from raw log files across supported datasets.

    Args:
        dir (str): Root directory containing raw log CSV files.
        save_dir (str): Destination directory for processed artifacts.
        dataset (str): Dataset name, one of GAIA or OpenRCA-Market.
        timezone (str): Timezone used to convert GAIA embedded timestamps to epoch ms.

    Returns:
        None
    """
    def get_log_files(dir, dataset):
        if dataset == "GAIA":
            log_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith("2021-07.csv")]
            logger.info(f"Found {len(log_files)} log files in {dir} (only data from 2021-07)")
        if "OpenRCA-Market" in dataset:
            log_files = glob.glob(os.path.join(dir, "*"))
            logger.info(f"Found {len(log_files)} log files in {dir}")
        return log_files
    
    def standardize_log_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Standardize log rows into a common schema for downstream processing.

        Args:
            df (pd.DataFrame): Input log data for a single file.
            dataset (str): Dataset key to control field mapping and timestamp handling.

        Returns:
            pd.DataFrame: Log rows with columns timestamp, entity, message, log_name, log_type.
        """
        if dataset == "GAIA":
            df = df.dropna(subset=['message'])
            df['timestamp'] = df['message'].map(lambda m: m.split(' |')[0].replace(",", "."))
            df['timestamp'] = df['timestamp'].apply(lambda x: convert_datetime_to_epoch(str(x), timezone))
            df = df.rename(columns={'service': 'entity'})
            df['log_name'] = None # 'default-gaia-log'
            df['log_type'] = None # 'service'  # assume GAIA logs are service-level
            df = df[['timestamp', 'entity', 'message', 'log_name', 'log_type']]
        elif "OpenRCA-Market" in dataset:
            df = df.rename(columns={
                'cmdb_id': 'entity',
                'value': 'message'
            })
            df = df.dropna(subset=['message'])
            # Timestamps are in seconds, convert to ms
            df['timestamp'] = df['timestamp'].astype(float) * 1000
            df['timestamp'] = df['timestamp'].astype(int)
            df['log_type'] = df['log_name'].apply(
                lambda x: 'proxy' if 'envoy' in x.lower() else 'service'
            )
            df = df[['timestamp', 'entity', 'message', 'log_name', 'log_type']]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return df

    logger.info("Processing logs")
    save_file = os.path.join(save_dir, "log.csv")
    
    log_files = get_log_files(dir, dataset)
    dfs = []
    for file_path in tqdm(log_files, desc="Loading log CSVs"):
        try:
            df = pd.read_csv(file_path, quotechar='"', engine='python')
            df = standardize_log_df(df, dataset)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    log_df = pd.concat(dfs) if dfs else pd.DataFrame(columns=['timestamp', 'entity', 'message', 'log_name', 'log_type'])
    
    # logger.info("Ordering by timestamp...")
    # log_df = log_df.sort_values(by="timestamp").reset_index()
    logger.info(f"Saving {len(log_df)} log results...")
    log_df.to_csv(save_file, index=False)
    logger.info(f"Saved processed logs to {save_file}.")

def extract_traces_pre_post_fault(trace_df: pd.DataFrame, start_time, pre_window=PRE_FAULT_WINDOW, post_window=POST_FAULT_WINDOW):
    """
    Slice traces into pre- and post-fault windows using span start times.

    Args:
        trace_df (pd.DataFrame): Processed traces with start_time column in ms.
        start_time (int): Fault start timestamp in ms.
        pre_window (int): Milliseconds before start_time to include.
        post_window (int): Milliseconds after start_time to include.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Pre-fault traces, Post-fault traces.
    """
    pre_trace = trace_df[(trace_df['start_time'] > start_time - pre_window) & (trace_df['start_time'] < start_time)]
    post_trace = trace_df[(trace_df['start_time'] > start_time) & (trace_df['start_time'] < start_time + post_window)]
    return pre_trace, post_trace

def extract_logs_pre_post_fault(log_df: pd.DataFrame, start_time, pre_window=PRE_FAULT_WINDOW, post_window=POST_FAULT_WINDOW):
    """
    Slice logs into pre- and post-fault windows using log timestamps.

    Args:
        log_df (pd.DataFrame): Processed logs with timestamp column in ms.
        start_time (int): Fault start timestamp in ms.
        pre_window (int): Milliseconds before start_time to include.
        post_window (int): Milliseconds after start_time to include.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Pre-fault logs, Post-fault logs.
    """
    pre_log = log_df[(log_df['timestamp'] > start_time - pre_window) & (log_df['timestamp'] < start_time)]
    post_log = log_df[(log_df['timestamp'] > start_time) & (log_df['timestamp'] < start_time + post_window)]
    return pre_log, post_log

def extract_metrics_pre_post_fault(metric_df: pd.DataFrame, start_time, pre_window=PRE_FAULT_WINDOW, post_window=POST_FAULT_WINDOW):
    """
    Slice metrics into pre- and post-fault windows using metric timestamps.

    Args:
        metric_df (pd.DataFrame): Processed metrics with timestamp column in ms.
        start_time (int): Fault start timestamp in ms.
        pre_window (int): Milliseconds before start_time to include.
        post_window (int): Milliseconds after start_time to include.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Pre-fault metrics, Post-fault metrics.
    """
    pre_metric = metric_df[(metric_df['timestamp'] > start_time - pre_window) & (metric_df['timestamp'] < start_time)]
    post_metric = metric_df[(metric_df['timestamp'] > start_time) & (metric_df['timestamp'] < start_time + post_window)]
    return pre_metric, post_metric

def get_all_entities_with_metrics(dataset: str):
    if dataset == "GAIA":
        return ["dbservice1", "dbservice2", 
                "mobservice1", "mobservice2", 
                "logservice1", "logservice2", #
                "webservice1", "webservice2", 
                "redisservice1", "redisservice2",
                "redis", "zookeeper",
                "system"]
    else:
        return []

def process_metrics_GAIA(dir, save_dir):
    """
    Merge GAIA metric CSVs into a single long-form parquet.

    Args:
        dir (str): Root directory containing GAIA metric CSVs.
        save_dir (str): Destination directory for processed artifacts.

    Returns:
        pd.DataFrame: Long-form metrics with entity, metric, timestamp, value.
    """
    logger.info(f"Reading all metrics from directory {dir}")
    save_file = os.path.join(save_dir, "metric.parquet")
    
    entities = get_all_entities_with_metrics("GAIA")
    metric_files = [f for f in os.listdir(dir) if "2021-08" not in f]
    logger.info(f"Found {len(metric_files)} metric files in {dir} (only data from 2021-07)")
    
    dfs = []
    for f in tqdm(metric_files, desc="Loading metric CSVs"):
        splits = f.split('_')
        entity, host = splits[0], splits[1]
        if (entity not in entities):
            continue
        metric_name = '_'.join(splits[2:-2])
        key = f"{entity}_{host}"
        
        df = pd.read_csv(os.path.join(dir, f))
        df['entity'] = key
        df['metric'] = metric_name
        dfs.append(df)
        
    metric_df = pd.concat(dfs)
    metric_df = metric_df.sort_values(by=['entity', 'metric', 'timestamp']).reset_index(drop=True)
    
    logger.info("Saving results...")
    metric_df.to_parquet(save_file, index=False)
    logger.info(f"Saved processed metrics to {save_file}.")
    
    return metric_df

def process_metrics_Market(dir, save_dir, oracle_kpis):
    """
    Merge OpenRCA-Market metric files into one parquet filtered by oracle KPIs.

    Args:
        dir (str): Root directory containing OpenRCA-Market metric CSVs.
        save_dir (str): Destination directory for processed artifacts.
        oracle_kpis (List[str]): KPIs to retain when building the long-form table.

    Returns:
        pd.DataFrame: Long-form metrics with entity, metric, timestamp, value.
    """
    logger.info(f"Reading all metrics from directory {dir}")
    save_file = os.path.join(save_dir, "metric.parquet")
    
    metric_files = glob.glob(os.path.join(dir, "*"))
    logger.info(f"Found {len(metric_files)} metric files in {dir}")
    
    all_rows = []
    for file_path in tqdm(metric_files, desc="Loading metric CSVs"):
        if file_path.endswith("metric_service.csv"):
            df = pd.read_csv(file_path)
            df = df[df['service'].isin(oracle_kpis)]
            # Convert timestamps to ms
            df['timestamp'] = df['timestamp'].astype(float) * 1000
            df['timestamp'] = df['timestamp'].astype(int)

            for _, row in df.iterrows():
                service = row['service']
                if '-' not in service:
                    continue
                base, proto = service.split('-')
                for col in ['rr', 'sr', 'mrt', 'count']:
                    kpi_name = f"{proto}-{col}"
                    value = row[col]
                    all_rows.append({
                        'entity': base,
                        'metric': kpi_name,
                        'timestamp': row['timestamp'],
                        'value': value
                    })
        else:
            df = pd.read_csv(file_path)
            df = df[df['kpi_name'].isin(oracle_kpis)]
            # Convert timestamps to ms
            df['timestamp'] = df['timestamp'].astype(float) * 1000
            df['timestamp'] = df['timestamp'].astype(int)

            for _, row in df.iterrows():
                all_rows.append({
                    'entity': row['cmdb_id'],
                    'metric': row['kpi_name'],
                    'timestamp': row['timestamp'],
                    'value': row['value']
                })

    # Convert to DataFrame and sort
    metric_df = pd.DataFrame(all_rows)
    metric_df = metric_df.sort_values(by=['entity', 'metric', 'timestamp']).reset_index(drop=True)

    logger.info("Saving results...")
    metric_df.to_parquet(save_file, index=False)
    logger.info(f"Saved processed metrics to {save_file}")

    return metric_df

def process_metrics(dataset, dir, save_dir, oracle_kpis) -> pd.DataFrame:
    """
    Dispatch to dataset-specific metric processors and return the resulting DataFrame.

    Args:
        dataset (str): Dataset key used to select GAIA or OpenRCA-Market*.
        dir (str): Root directory containing raw metric files.
        save_dir (str): Destination directory for processed parquet.
        oracle_kpis (List[str]): KPI names to retain where applicable.

    Returns:
        pd.DataFrame: Long-form metrics with entity, metric, timestamp, value.
    """
    if dataset == "GAIA":
        metrics_df = process_metrics_GAIA(dir, save_dir)
    elif "OpenRCA-Market" in dataset:
        metrics_df = process_metrics_Market(dir, save_dir, oracle_kpis)
    return metrics_df

def compute_pre_post_fault_windows(label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute safe pre and post fault window durations per fault avoiding temporal overlap with other faults.
    Assumes label_df is sorted by 'fault_st_time'.

    Args:
        label_df (pd.DataFrame): Labels containing fault_id, fault_st_time, fault_ed_time in ms.

    Returns:
        pd.DataFrame: Input with added pre_window and post_window columns in ms.
    """
    label_df = label_df.sort_values(by='fault_st_time').reset_index(drop=True)
    pre_windows, post_windows = [], []
    
    for i, row in label_df.iterrows():
        fault_id = row['fault_id']
        st_time = row['fault_st_time']
        ed_time = row['fault_ed_time']

        # Pre-fault window: No fault should be active in [st_time - pre_window, st_time)
        pre_window_start = 0 # assume time starts at 0
        for j in range(i): # only look at earlier faults
            other_st = label_df.loc[j, 'fault_st_time']
            other_ed = label_df.loc[j, 'fault_ed_time']
            
            # If the other fault overlaps with [?, st_time), restrict the pre-window
            if (other_ed > pre_window_start) and (other_st < st_time):
                pre_window_start = max(pre_window_start, other_ed + MIN_BUFFER)
        pre_window = max(0, st_time - pre_window_start)

        # Post-fault window: No other fault should *start* in [st_time, st_time + post_window]
        if i < len(label_df) - 1:
            next_st_time = label_df.loc[i + 1, 'fault_st_time']
            post_window = min(ed_time, next_st_time) - st_time
        else:
            post_window = ed_time - st_time  # last fault in list
        post_window = max(0, post_window)  # safety
        
        pre_windows.append(pre_window)
        post_windows.append(post_window)

    label_df['pre_window'] = pre_windows
    label_df['post_window'] = post_windows
    return label_df
    
def build_pre_post_data(label_df: pd.DataFrame, trace_df: pd.DataFrame, log_df: pd.DataFrame, metric_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Construct pre- and post-fault telemetry per fault using computed windows.

    Args:
        label_df (pd.DataFrame): Labels with fault_st_time and fault_ed_time in ms.
        trace_df (pd.DataFrame): Processed trace table.
        log_df (pd.DataFrame): Processed log table.
        metric_df (pd.DataFrame): Processed metric table.

    Returns:
        Tuple[Dict, Dict]: Two dictionaries keyed by fault_id with trace, log, metric DataFrames.
    """
    label_df = compute_pre_post_fault_windows(label_df)

    pre_fault_data, post_fault_data = {}, {}
        
    def _build_single_fault_window(row):
        fault_id = row['fault_id']
        start_time = row['fault_st_time']
        pre_window = row['pre_window']
        post_window = row['post_window']

        pre_trace, post_trace = extract_traces_pre_post_fault(trace_df, start_time, pre_window, post_window)
        pre_log, post_log = extract_logs_pre_post_fault(log_df, start_time, pre_window, post_window)
        pre_metric, post_metric = extract_metrics_pre_post_fault(metric_df, start_time, pre_window, post_window)

        return fault_id, {
            'trace': pre_trace,
            'log': pre_log,
            'metric': pre_metric
        }, {
            'trace': post_trace,
            'log': post_log,
            'metric': post_metric
        }
        
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(_build_single_fault_window, row): row['fault_id']
            for _, row in label_df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Building pre- and post-data per fault (in parallel)"):
            fault_id, pre, post = future.result()
            pre_fault_data[fault_id] = pre
            post_fault_data[fault_id] = post

    return pre_fault_data, post_fault_data

def build_normal_data(start: int, end: int, trace_df: pd.DataFrame, log_df: pd.DataFrame, metric_df: pd.DataFrame) -> Dict:
    """
    Extract a normal baseline window for logs, traces, and metrics.

    Args:
        start (int): Start of the normal window in ms.
        end (int): End of the normal window in ms.
        trace_df (pd.DataFrame): Processed trace table.
        log_df (pd.DataFrame): Processed log table.
        metric_df (pd.DataFrame): Processed metric table.

    Returns:
        Dict: Dictionary with keys log, trace, metric mapped to DataFrames in the window.
    """
    normal_data = {}
    
    normal_logs = log_df[(log_df['timestamp'] >= start) & (log_df['timestamp'] <= end)]
    normal_traces = trace_df[(trace_df['start_time'] >= start) & (trace_df['timestamp'] <= end)]
    normal_metrics = metric_df[(metric_df['timestamp'] >= start) & (metric_df['timestamp'] <= end)]
    
    normal_data['log'] = normal_logs
    normal_data['trace'] = normal_traces
    normal_data['metric'] = normal_metrics
    
    return normal_data
    
def main():
    """
    CLI entry point: process or load artifacts, compute windows, and export slices.

    Loads config, builds processed trace/log/metric artifacts if missing, normalizes
    label timestamps, computes safe pre and post windows, and writes out the
    pre-fault-data, post-fault-data, and optional normal-data directories under processed-data.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Pre-processes logs, metrics, and traces into pre- and post-failure segments.")
    parser.add_argument("--config", type=str, default="code/extractor/config.yaml", help="Path to config file.")
    parser.add_argument("--dataset", type=str, default="GAIA", help="Dataset to process.")
    args = parser.parse_args()
    
    config = io_util.load_config(args.config)
    dataset = args.dataset
    dataset_config = config[dataset]

    timezone = dataset_config['timezone']
    path_to_dataset = dataset_config['telemetry-data']
    processed_dir = dataset_config['processed-data']
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    logger.info(f"Starting pre- and post-fault telemetry extraction for {dataset}")
    logger.info(f"Using timezone: {timezone}")
    
    try:
        trace_df = pd.read_csv(os.path.join(processed_dir, "trace.csv"), dtype={"parent_span_id": str})
        logger.info(f"Loaded processed traces from {os.path.join(processed_dir, 'trace.csv')}")
    except Exception as e:
        logger.warning(f"Failed to load processed trace CSV: {e}")
        logger.info(f"Processing raw traces...")
        st = time.time()
        process_traces(
            os.path.join(path_to_dataset, dataset_config['traces-path']), 
            processed_dir,
            dataset,
            timezone
        )
        logger.info(f"Finished processing traces. Time taken: {time.time() - st}")
        trace_df = pd.read_csv(os.path.join(processed_dir, "trace.csv"), dtype={"parent_span_id": str})
    print(trace_df.head(10))
    
    try:
        log_df = pd.read_csv(os.path.join(processed_dir, "log.csv"))
        logger.info(f"Loaded processed logs from {os.path.join(processed_dir, 'log.csv')}")
    except Exception as e:
        logger.warning(f"Failed to load processed log CSV: {e}")
        logger.info(f"Processing raw logs...")
        st = time.time()
        process_logs(
            os.path.join(path_to_dataset, dataset_config['logs-path']), 
            processed_dir,
            dataset,
            timezone
        )
        logger.info(f"Finished processing logs. Time taken: {time.time() - st}")
        log_df = pd.read_csv(os.path.join(processed_dir, "log.csv"))
    print(log_df.head(10))
    
    try:
        metric_df = pd.read_parquet(os.path.join(processed_dir, "metric.parquet"))
        logger.info(f"Loaded processed metrics from {os.path.join(processed_dir, 'metric.parquet')}")
    except Exception as e:
        logger.warning(f"Failed to load processed metric parquet: {e}")
        logger.info(f"Processing raw metrics...")
        st = time.time()
        process_metrics(
            dataset,
            os.path.join(path_to_dataset, dataset_config['metrics-path']), 
            processed_dir,
            dataset_config['oracle-kpis']
        )
        logger.info(f"Finished processing metrics. Time taken: {time.time() - st}")
        metric_df = pd.read_parquet(os.path.join(processed_dir, "metric.parquet"))
    print(metric_df.head(10))
    
    logger.info("Sorting the data by time for more efficient indexing...")
    trace_df = trace_df.sort_values(by='start_time').reset_index()
    log_df = log_df.sort_values(by='timestamp').reset_index()
    metric_df = metric_df.sort_values(by='timestamp').reset_index()
    
    logger.info("Loading the label files...")
    # Use the pre-filtered fault injection "run" table (if available) to accurately extract "normal" non-fault data
    if dataset_config.get('run-table', False):
        label_df = pd.read_csv(dataset_config['run-table'])
    else:
        label_df = pd.read_csv(dataset_config['gt-file'])
    
    logger.info("Standardizing time formats in label file...")
    if dataset == 'GAIA':
        label_df['fault_st_time'] = label_df['fault_st_time'].apply(lambda x: convert_datetime_to_epoch(str(x), timezone))
        label_df['fault_ed_time'] = label_df['fault_ed_time'].apply(lambda x: convert_datetime_to_epoch(str(x), timezone))
    elif ("OpenRCA-Market" in dataset):
        # Convert timestamps and durations to ms
        label_df['fault_st_time'] = label_df['fault_st_time'].apply(lambda x: int(x) * 1000)
        label_df['fault_ed_time'] = label_df['fault_ed_time'].apply(lambda x: int(x) * 1000)
        label_df['fault_duration'] = label_df['fault_duration'].apply(lambda x: int(x) * 1000)

    logger.info("Starting pre-/post-fault data building...")
    # Get pre-injected fault data and post-injected fault data.
    # Note: pre_fault_data should include non-anomalous/normal signals. post_fault_data should include anomalous signals.
    # Indexes of pre and post-fault data correspond to the run_table, if it exists
    pre_fault_data, post_fault_data = build_pre_post_data(label_df, trace_df, log_df, metric_df)
    
    if dataset_config.get('normal-period', False):
        logger.info(f"Using normal data from {dataset_config['normal-period']['start']} to {dataset_config['normal-period']['end']}")
        logger.info("Building normal data...")
        normal_data = build_normal_data(
            dataset_config['normal-period']['start'],
            dataset_config['normal-period']['end'],
            trace_df, log_df, metric_df)
    else:
        normal_data = None

    logger.info("Saving pre-fault-data...")
    io_util.save_fault_data(
        os.path.join(processed_dir, "pre-fault-data"), 
        pre_fault_data)

    logger.info("Saving post-fault-data...")
    io_util.save_fault_data(
        os.path.join(processed_dir, "post-fault-data"), 
        post_fault_data)
    logger.info(f"Results (pre-fault-data and post-fault-data) saved to {processed_dir}")
    
    if normal_data:
        logger.info("Saving normal data to normal-data")
        io_util.save_normal_data(
            os.path.join(processed_dir, "normal-data"), 
            normal_data)
        logger.info(f"Results (normal-data) saved to {processed_dir}")
    
    logger.info(f"Completed processing.")

# python extract_pre_and_post_fault_telemetry.py --config code/extractor/config.yaml --dataset GAIA
if __name__ == "__main__":
    main()