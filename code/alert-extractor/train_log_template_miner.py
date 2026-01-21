"""
Extract Log Templates with Drain3

Purpose:
- Load post-fault logs for a dataset and extract stable templates with Drain3
- Persist the trained miner state and export a template frequency summary CSV

Prerequisites:
- gt-file points to the label.csv produced by the dataset prep script
    (prep_GAIA_dataset.py, prep_OpenRCA_Market_dataset.py). 
    If run-table is present, labels are indexed by orig_fault_id.
- processed-data/post-fault-data exists (built by extract_pre_and_post_fault_telemetry.py)
- fault-alert-data/log directory exists and is writable (output location)
- system is set in config.yaml (used for the drain3-<system>-state.bin filename)

Workflow:
1) Load dataset configuration, labels, and post-fault log data
2) Flatten raw log messages across selected faults
3) Train or update the Drain3 TemplateMiner and persist its state
4) Save a CSV summary of templates with id, template text, and count

Outputs:
- fault-alert-data/log/drain3-<system>-state.bin persisted for reuse
- fault-alert-data/log/drain-templates.csv with columns [id, template, count]

Examples:
    # python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset GAIA
    # python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-1"
    # python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-2"
"""
import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils.log_util import get_logger

from drain.drain_template_extractor import TemplateMiner, extract_log_templates
from utils import io_util

logger = get_logger(__name__)

def extract_training_logs(label_df: pd.DataFrame, post_data: dict) -> list[str]:
    """
    Extract post-fault raw log messages for the faults present in label_df.

    Expects post_data keyed by fault id and containing a log dataframe with a
    message column. Iterates the label_df index and collects all available
    messages for corresponding faults.

    Args:
        label_df (pd.DataFrame): Fault label table; its index provides fault ids.
        post_data (dict): Mapping fault id to telemetry dict with key 'log'.

    Returns:
        list[str]: All log messages from post-fault training windows.
    """
    logs = []
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Extracting training logs"):
        log_chunk = post_data[idx]['log']
        logs.extend(log_chunk['message'].dropna().astype(str).tolist())
    return logs


def build_template_dataframe(miner: TemplateMiner) -> pd.DataFrame:
    """
    Build a dataframe summarizing log templates from a trained Drain3 TemplateMiner.

    Sorts clusters by frequency and emits id, template text, and count columns.

    Args:
        miner (TemplateMiner): Trained TemplateMiner instance.

    Returns:
        pd.DataFrame: Template summary with columns id, template, count.
    """
    sorted_clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=True)
    return pd.DataFrame({
        'id': [c.cluster_id for c in sorted_clusters],
        'template': [c.get_template() for c in sorted_clusters],
        'count': [c.size for c in sorted_clusters]
    })


def main():
    """
    CLI entry point to extract log templates using Drain3.
    - Loads dataset configuration and labels,
    - retrieves post-fault logs, 
    - trains a TemplateMiner with all messages,
    - persists its state, 
    - and writes a template summary CSV to the fault alert directory.
    """
    parser = argparse.ArgumentParser(description="Extract log templates using Drain.")
    parser.add_argument("--config", type=str, default="code/alert-extractor/config.yaml", help="Path to config file.")
    parser.add_argument("--dataset", type=str, default="OpenRCA-Market-cloudbed-1", help="Dataset to process.")
    args = parser.parse_args()

    # Load dataset configuration
    config = io_util.load_config(args.config)
    dataset = args.dataset
    dataset_config = config[dataset]

    dataset_dir = dataset_config['telemetry-data']
    processed_dir = dataset_config['processed-data']
    fault_alert_dir = dataset_config['fault-alert-data']

    logger.info(f"Starting log template extraction for {dataset}")
    logger.info("Loading fault labels and post-fault logs...")
    if dataset_config.get('run-table', False):
        # label_df = pd.read_csv(dataset_config['run-table'], index_col='fault_id') # Use all faults
        label_df = pd.read_csv(dataset_config['gt-file'], index_col='orig_fault_id') # Only use selected faults -- need orig_fault_id for proper indexing
    else:
        label_df = pd.read_csv(dataset_config['gt-file'], index_col='fault_id')
        
    post_data = io_util.load_fault_data(
        os.path.join(processed_dir, 'post-fault-data'),
        modalities_to_skip=['metric', 'trace'],
        fault_ids_to_include=list(label_df.index)
    )
    
    logs = extract_training_logs(label_df, post_data)

    logger.info(f"Extracting {len(logs)} log messages into templates...")
    miner = extract_log_templates(
        dataset_config['system'],
        logs,
        os.path.join(fault_alert_dir, 'log')
    )

    logger.info("Saving template summary...")
    template_df = build_template_dataframe(miner)
    template_df.to_csv(
        os.path.join(fault_alert_dir, 'log/drain-templates.csv'), 
        index=False
    )
    logger.info("Log template extraction complete.")

# python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset GAIA
# python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-1"
# python code/alert-extractor/train_log_template_miner.py --config code/alert-extractor/config.yaml --dataset "OpenRCA-Market-cloudbed-2"
if __name__ == "__main__":
    main()