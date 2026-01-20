"""
Prepare GAIA Dataset for Alert Extraction and Evaluation

Purpose:
- Parse and normalize injected fault metadata from the original run table
- Identify telemetry gaps across metrics, traces, and logs to filter unreliable periods
- Filter overlapping faults and balance the dataset, augmenting rare fault types

Prerequisites:
- GAIA dataset is already downloaded locally
- The alert-extractor config.yaml is updated so 'telemetry-data' and 'processed-data' point to local dataset directories

Workflow:
1) Format the original run table into a consistent fault label file
2) Scan telemetry files to find large time gaps per service
3) Filter faults for temporal overlap and for telemetry gaps
4) Balance and augment rare fault types, writing the final label.csv

Outputs:
- run_table.csv, telemetry_gaps.json, filtered_overlap_run_table.csv, filtered_run_table.csv
- label.csv containing balanced and augmented GAIA fault samples
"""
import os
import json
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
from utils.time_util import convert_datetime_to_epoch, convert_epoch_to_datetime_str
from utils.io_util import load_config
from utils.log_util import get_logger

RARE_FAULT_TYPES = [
    "[access permission denied exception]",
    "[file moving program]",
    "[normal memory freed label]",
]
PRIORITY_FAULT_TYPES = RARE_FAULT_TYPES + ["[memory_anomalies]"]

TIME_GAP_THRESHOLD = 30 * 60 * 1000  # 30 minutes in milliseconds
FAULT_SEPARATION_WINDOW = 45 * 1000 # 45 seconds in milliseconds

logger = get_logger(__name__)

def format_run_table(run_table_file, save_file):
    """
    Extract and properly format injected fault information from GAIA's run table.

    Parses known fault messages to derive start time, end time, duration, type,
    and normalizes fields for downstream processing. Writes a sorted, filtered
    CSV with fault-level metadata and a service_instance fault_level.

    Args:
        run_table_file (str): Path to the original run table CSV.
        save_file (str): Output path for the formatted fault CSV.
    """

    def extract_injected_fault_info(row):
        message = str(row['message']).rstrip('\n')
        if "[memory_anomalies]" in message:
            match = re.search(r'start at ([\d\-:\. ]+)', message)
            if not match:
                logger.warning("no match for memory anomaly")
            start_time_str = match.group(1)
            time_diff = 600
            fault_type = "[memory_anomalies]"
        elif "[normal memory freed label]" in message:
            start_time_str = message.split(" |")[0].replace(",", ".")
            time_diff = 600
            fault_type = "[normal memory freed label]"
        elif "login failure" in message:
            start_time_str = message.split(" |")[0].replace(",", ".")
            time_diff = 11
            fault_type = "[login failure]"
        elif "file moving program" in message:
            match = re.search(r'start with ([\d\-:\. ]+)', message)
            if not match:
                logger.warning("no match for file moving program")
            start_time_str = match.group(1)
            time_diff = 600
            fault_type = "[file moving program]"
        elif "access permission denied" in message:
            start_time_str = message.split(" |")[0].replace(",", ".")
            time_diff = 3600
            fault_type = "[access permission denied exception]"
        else:
            start_time_str = None
            
        if start_time_str:
            try:
                start_time = datetime.strptime(start_time_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                # Handle cases where milliseconds are not present
                start_time = datetime.strptime(start_time_str.strip(), "%Y-%m-%d %H:%M:%S")
            end_time = start_time + timedelta(seconds=time_diff)
        else:
            start_time = None
            end_time = None
            time_diff = None
            fault_type = None
            
        row['fault_type'] = fault_type
        row['fault_st_time'] = start_time
        row['fault_ed_time'] = end_time
        row['fault_duration'] = time_diff
        row['message'] = message
        return row
            
    run_table = pd.read_csv(run_table_file)
    run_table = run_table.apply(extract_injected_fault_info, axis=1)
    run_table.dropna(subset=['fault_st_time'], inplace=True)
    run_table.sort_values(by="fault_st_time", inplace=True)
    run_table.rename(columns={"service": "fault_entity", "message": "fault_message"}, inplace=True)
    run_table["fault_level"] = "service_instance"
    run_table.drop(columns='datetime', inplace=True)

    run_table = run_table[['fault_level', 'fault_entity', 'fault_type', 'fault_st_time', 'fault_ed_time', 'fault_duration', 'fault_message']].reset_index(drop=True)
    run_table.index.name  = 'fault_id'
    run_table.to_csv(save_file, index=True)

def extract_telemetry_gaps(metrics_dir, traces_dir, logs_dir, save_file):
    """
    Find time gaps in telemetry data across metrics, traces, and logs.

    Examines per-file timestamps, inserts boundary timestamps, and records
    service-specific gaps exceeding TIME_GAP_THRESHOLD. Produces a JSON with
    (gap_start, gap_end) keys mapped to the affected services and files.

    Args:
        metrics_dir (str): Directory containing metric CSVs.
        traces_dir (str): Directory containing trace CSVs.
        logs_dir (str): Directory containing log CSVs.
        save_file (str): Output path for telemetry gaps JSON.
    """
    gap_groups = defaultdict(dict)
    
    def find_time_gaps(data_type, dir, time_start, time_cutoff, file_name_exclusions):
        def extract_service_name(filename, data_type):
            if data_type == 'metric':
                return filename.split("_")[0]
            elif data_type == 'trace':
                return filename.split("_")[2]
            elif data_type == 'log':
                return filename.split("_")[2]
        
        for filename in tqdm(os.listdir(dir), desc=f"Processing files in {dir}"):
            if not filename.endswith(".csv") or any(excl in filename for excl in file_name_exclusions):
                continue

            filepath = os.path.join(dir, filename)
            try:
                if data_type in ['metric', 'trace']:
                    df = pd.read_csv(filepath, usecols=["timestamp"])
                    df = df.dropna().sort_values("timestamp")
                elif data_type == 'log':
                    df = pd.read_csv(filepath, header=0)
                    
                if data_type == 'metric':
                    timestamps = df["timestamp"].astype(int).values
                elif data_type == 'trace':
                    timestamps = df["timestamp"].apply(lambda x: convert_datetime_to_epoch(str(x), "Asia/Shanghai")).to_list()
                elif data_type == 'log':
                    timestamps = df["message"].dropna().apply(lambda x: convert_datetime_to_epoch(str(x).split(" |")[0].replace(",", "."), "Asia/Shanghai")).to_list()

                if time_start not in timestamps:
                    timestamps = np.insert(timestamps, 0, time_start)
                if time_cutoff not in timestamps:
                    timestamps = np.append(timestamps, time_cutoff)
                    
                for i in range(1, len(timestamps)):
                    start, end = timestamps[i - 1], timestamps[i]
                    if (end - start) > TIME_GAP_THRESHOLD:
                        service = extract_service_name(filename, data_type)
                        if service not in gap_groups[(start, end)]:
                            gap_groups[(start, end)][service] = []
                        gap_groups[(start, end)][service].append(f'{data_type}/{filename}')
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

    first_fault_time = convert_datetime_to_epoch("2021-07-01 11:06:52.249428", "Asia/Shanghai") # timestamp of first injected fault
    time_mid_point = convert_datetime_to_epoch("2021-07-14 23:59:59.999999", "Asia/Shanghai") # July 14 at 11:59 PM = 1626278399999
    time_cutoff = convert_datetime_to_epoch("2021-07-31 23:59:59.999999", "Asia/Shanghai") # end of 07-31
    
    find_time_gaps('metric', metrics_dir, first_fault_time, time_mid_point, ['2021-08', '2021-07-31']) # gaps for 07-01 to 07-15
    find_time_gaps('metric', metrics_dir, time_mid_point, time_cutoff, ['2021-08', '2021-07-01']) # gaps for 07-15 to 07-31
    find_time_gaps('trace', traces_dir, first_fault_time, time_cutoff, ['2021-08'])
    find_time_gaps('log', logs_dir, first_fault_time, time_cutoff, ['2021-08'])

    # Sort gap_groups keys based on the start timestamp (first item in the tuple)
    sorted_gap_groups = dict(sorted(gap_groups.items(), key=lambda x: x[0][0]))

    # Convert tuple keys to strings for JSON serialization
    gap_groups_serializable = {str(key): value for key, value in sorted_gap_groups.items()}

    with open(save_file, 'w') as f:
        json.dump(gap_groups_serializable, f, indent=2)

def filter_faults_for_overlap(run_table_file, save_file, fault_separation_window=FAULT_SEPARATION_WINDOW):
    """
    Filter faults to avoid temporal overlap within a separation window.

    Converts times to epoch, iterates faults ordered by start time, and applies
    rules to keep non-overlapping samples. Outputs a filtered CSV with readable
    datetime strings.

    Args:
        run_table_file (str): Input formatted run_table.csv.
        save_file (str): Output path for filtered CSV.
        fault_separation_window (int): Minimum separation in ms between faults.
    """
    run_table = pd.read_csv(run_table_file)
    logger.info(f"Original samples: {len(run_table)}")
    logger.info(run_table['fault_type'].value_counts())
    
    run_table['fault_st_time'] = run_table['fault_st_time'].apply(lambda x: convert_datetime_to_epoch(x, timezone='Asia/Shanghai'))
    run_table['fault_ed_time'] = run_table['fault_ed_time'].apply(lambda x: convert_datetime_to_epoch(x, timezone='Asia/Shanghai'))

    fault_rows = run_table.to_dict("records")
    selected_rows = []
    latest_end_time = None # *Latest* fault_ed_time seen so far
    
    for i, row in enumerate(fault_rows):
        if i == 0:
            # First row: always keep
            selected_rows.append(row)
            prev_fault = row
            latest_end_time = row['fault_ed_time']
            continue
        
        prev_fault = fault_rows[i-1]
        
        # Define time-based overlap conditions relative to prev_fault
        overlaps_temporally = row['fault_st_time'] < prev_fault['fault_ed_time']
        sufficiently_separated = row['fault_st_time'] >= (prev_fault['fault_st_time'] + fault_separation_window)
        
        if not overlaps_temporally:
            if selected_rows and (latest_end_time < row['fault_st_time']):
                selected_rows.append(row)
        elif sufficiently_separated:
            if selected_rows and (latest_end_time < row['fault_st_time']):
                selected_rows.append(row)
        elif not sufficiently_separated:
            if selected_rows and selected_rows[-1]['fault_id'] == prev_fault['fault_id']:
                selected_rows.pop() # Remove previous kept 
        else:
            pass
        if row['fault_ed_time'] > latest_end_time:
            latest_end_time = row['fault_ed_time']
            
    filtered_df = pd.DataFrame(selected_rows)
    filtered_df.set_index('fault_id', inplace=True)
    filtered_df['fault_st_time'] = filtered_df['fault_st_time'].apply(lambda x: convert_epoch_to_datetime_str(x, timezone='Asia/Shanghai'))
    filtered_df['fault_ed_time'] = filtered_df['fault_ed_time'].apply(lambda x: convert_epoch_to_datetime_str(x, timezone='Asia/Shanghai'))
    filtered_df.to_csv(save_file)
    
    logger.info(f"Filtered for temporal overlap with window {fault_separation_window} ms.")
    logger.info(f"Remaining samples: {len(filtered_df)}")
    logger.info(filtered_df['fault_type'].value_counts())
    
def filter_faults_for_telemetry_gaps(run_table_file, telemetry_gaps_file, save_file):
    """
    Filter injected faults that overlap with detected telemetry gaps.

    Loads gap windows and removes faults whose intervals intersect with any
    service-level gap window (excluding system-only and zookeeper-only gaps).

    Args:
        run_table_file (str): Input CSV of faults indexed by fault_id.
        telemetry_gaps_file (str): JSON with gap windows and affected services.
        save_file (str): Output path for filtered CSV.

    Returns:
        None
    """
    run_table = pd.read_csv(run_table_file, index_col='fault_id')
    with open(telemetry_gaps_file, "r") as f:
        telemetry_gaps_serialized = json.load(f)
        
    # Convert string keys back to tuples
    telemetry_gaps = {
        eval(key): value for key, value in telemetry_gaps_serialized.items()
    }
    # Drop keys from telemetry_gaps where the only key in the value is "system" or "zookeeper"
    filtered_gaps = {
        key: value for key, value in telemetry_gaps.items() if list(value.keys()) != ["system"] and list(value.keys()) != ["zookeeper"]
    }

    # Filter out faults that overlap with telemetry gaps
    filtered_run_table = run_table.copy()

    for (gap_start, gap_end), _ in tqdm(filtered_gaps.items()):
        gap_start_epoch = gap_start
        gap_end_epoch = gap_end

        # Filter out rows where fault overlaps with the telemetry gap
        filtered_run_table = filtered_run_table[
            ~(
                (filtered_run_table['fault_st_time'].apply(lambda x: convert_datetime_to_epoch(x, timezone='Asia/Shanghai')) <= gap_end_epoch) &
                (filtered_run_table['fault_ed_time'].apply(lambda x: convert_datetime_to_epoch(x, timezone='Asia/Shanghai')) >= gap_start_epoch)
            )
        ]

    # Save the filtered run_table
    filtered_run_table.to_csv(save_file, index=True)
    
    logger.info("Filtered sample distribution:")
    print(filtered_run_table['fault_type'].value_counts())
    print(filtered_run_table.groupby(['fault_type', 'fault_entity']).agg(count=('fault_type', 'count')))
    logger.info(f"Total samples: {len(filtered_run_table)}")
    logger.info(f"Total samples filtered out: {len(run_table) - len(filtered_run_table)}")

def augment_rare_fault_samples(rare_df, config) -> pd.DataFrame:
    """
    Augment rare faults by duplicating with shifted time windows.

    Duplicates each rare fault sample augmentations_per_fault times and applies
    time transformations to fault_st_time and fault_ed_time, marking samples as
    augmented.

    Args:
        rare_df (pd.DataFrame): Rare fault samples to augment.
        config (dict): Includes augmentations-per-fault, augmented-faults-st-time,
            augmented-faults-spacing, timezone.

    Returns:
        pd.DataFrame: Augmented samples with updated times and augmented flag.
    """
    
    augmentations_per_fault = config['augmentations-per-fault'] 
    augmented_faults_st_time = config['augmented-faults-st-time'] # Start time for augmented faults
    aug_inter_fault_gap = config['augmented-faults-spacing'] # Gap between augmented fault ed time and next augmented fault st time

    # Duplicate rare_df `augmentations_per_fault` times.
    rare_df_augmented_samples = pd.concat(
        [rare_df.copy() for _ in range(augmentations_per_fault)],
        ignore_index=False
    )
    
    # Apply time transformations to fault_st_time and fault_ed_time for each augmented sample.
    current_time = augmented_faults_st_time
    augmented_rows = []
    for _, row in rare_df_augmented_samples.iterrows():
        st_time = current_time
        ed_time = st_time + (row['fault_duration'] * 1000)

        new_row = row.copy()
        new_row['fault_st_time'] = convert_epoch_to_datetime_str(st_time, config['timezone'])
        new_row['fault_ed_time'] = convert_epoch_to_datetime_str(ed_time, config['timezone'])
        new_row['augmented'] = True

        augmented_rows.append(new_row)
        current_time = ed_time + aug_inter_fault_gap

    rare_df_augmented_samples = pd.DataFrame(augmented_rows)
    return rare_df_augmented_samples
    
def balance_and_augment_dataset(filtered_run_table_file, save_file, config):
    """
    Balance the dataset and augment rare fault types for GAIA.

    Samples memory anomalies and login failures per fault_entity, concatenates
    with rare and augmented samples, and writes the final label file.

    Args:
        filtered_run_table_file (str): Input CSV after telemetry-gap filtering.
        save_file (str): Output label.csv path.
        config (dict): Augmentation and timezone settings.
    """

    filtered_run_table = pd.read_csv(filtered_run_table_file, index_col="fault_id")

    rare_df = filtered_run_table[filtered_run_table['fault_type'].isin(RARE_FAULT_TYPES)]
    rare_augmented_samples_df = augment_rare_fault_samples(rare_df, config)

    # Sample per fault_entity for [memory_anomalies]
    memory_df = (
        filtered_run_table[
            filtered_run_table['fault_type'] == "[memory_anomalies]"
        ]
        .groupby('fault_entity', group_keys=False)
        .apply(
            lambda group: (
                group
                .assign(fault_entity=group.name)
                .sample(n=min(4, len(group)), random_state=42)
            ),
            include_groups=False,
        )
        .sort_values(by='fault_st_time')
    )

    # Sample per fault_entity for [login failure]
    login_df = (
        filtered_run_table[
            filtered_run_table['fault_type'] == "[login failure]"
        ]
        .groupby('fault_entity', group_keys=False)
        .apply(
            lambda group: (
                group
                .assign(fault_entity=group.name)
                .sample(n=min(20, len(group)), random_state=42)
            ),
            include_groups=False,
        )
        .sort_values(by='fault_st_time')
    )

    memory_login_df = pd.concat([memory_df, login_df])

    # Combine all samples
    final_eval_df = (
        pd.concat([rare_df, rare_augmented_samples_df, memory_login_df])
        .sort_values('fault_st_time')
        .rename_axis('orig_fault_id')
        .reset_index()
    )
    final_eval_df.index.name = 'fault_id'
    final_eval_df['augmented'] = final_eval_df['augmented'].convert_dtypes().fillna(False)

    final_eval_df.to_csv(save_file, index=True)

    logger.info("Final sample distribution:")
    print(final_eval_df['fault_type'].value_counts())
    print(final_eval_df.groupby(['fault_type', 'fault_entity']).agg(count=('fault_type', 'count')))
    logger.info(f"Total samples: {len(final_eval_df)}")

def main():
    """
    CLI entry point to prepare the GAIA dataset for alert extraction.
    - Executes standardized formatting of run table, 
    - telemetry gap detection,
    - fault overlap filtering,
    - telemetry-gap filtering, and 
    - dataset balancing/augmentation. 
    Writes outputs to the GAIA fault alert directory.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', default="code/extractor/config.yaml", type=str, help="Path to config file.")

    args = parser.parse_args()
    config = load_config(args.config)["GAIA"]

    dataset_dir = config['telemetry-data']
    processed_dir = config['processed-data']
    fault_alert_dir = config['fault-alert-data']
    
    metrics_dir = os.path.join(dataset_dir, config['metrics-path'])
    traces_dir = os.path.join(dataset_dir, config['traces-path'])
    logs_dir = os.path.join(dataset_dir, config['logs-path'])
    original_run_table_file = os.path.join(dataset_dir, config['original-run-table'])
    
    save_dir = fault_alert_dir
    run_table_file = os.path.join(save_dir, "run_table.csv")
    telemetry_gaps_file = os.path.join(save_dir, "telemetry_gaps.json")
    filtered_overlap_run_table_file = os.path.join(save_dir, "filtered_overlap_run_table.csv")
    filtered_run_table_file = os.path.join(save_dir, "filtered_run_table.csv")
    
    label_file = os.path.join(save_dir, "label.csv")
    
    logger.info("Prepping GAIA dataset...")
    logger.info(f"(1) Formatting the original run table: {original_run_table_file}")
    format_run_table(original_run_table_file, run_table_file)
    
    logger.info(f"(2) Extracting telemetry gaps of more than {TIME_GAP_THRESHOLD} milliseconds")
    extract_telemetry_gaps(metrics_dir, traces_dir, logs_dir, telemetry_gaps_file)
    
    logger.info(f"(3) Filtering run table to exclude faults injected with {FAULT_SEPARATION_WINDOW} milliseconds of each other")
    filter_faults_for_overlap(run_table_file, filtered_overlap_run_table_file)
    
    logger.info("(4) Filtering run table according to the detected telemetry gaps")
    filter_faults_for_telemetry_gaps(filtered_overlap_run_table_file, telemetry_gaps_file, filtered_run_table_file)
    
    logger.info("(5) Balancing the dataset")
    balance_and_augment_dataset(filtered_run_table_file, label_file, config)
    
    logger.info("GAIA dataset prep complete.")

if __name__ == "__main__":
    main()