"""
Drain3 Template Extraction Utilities

Purpose:
- Initialize a Drain3 `TemplateMiner` with system-specific configuration and on-disk persistence
- Train the miner over raw log lines to extract stable log templates
- Persist trained state for reuse across runs and print templates by frequency

Notes:
- Configuration is loaded from `drain3-{system}.ini` alongside this module
- Miner state is persisted to `drain3-{system}-state.bin` via `FilePersistence`
- Based on Drain3 (IBM): https://github.com/IBM/Drain3

"""
import os
import logging
from tqdm import tqdm

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

KEEP_TOP_N_TEMPLATE = 1000

def init_drain(system, save_file) -> TemplateMiner:
    """
    Initialize a Drain3 `TemplateMiner` with system-specific config (from .ini file) and file persistence.

    Loads `drain3-{system}.ini` from the current directory and sets up a
    `FilePersistence` at `save_file`. If the persistence file exists, the miner
    will load previous state from disk on initialization.

    Args:
        system (str): System identifier used to select the ini file.
        save_file (str): Path to the persistence file to store/load miner state.

    Returns:
        TemplateMiner: Configured miner ready to ingest log lines.
    """
    # Load config
    config = TemplateMinerConfig()
    config_path = os.path.join(
        os.path.dirname(__file__),
        f"drain3-{system}.ini"
    )
    config.load(config_path)
    config.profiling_enabled = True
    config.profiling_report_sec = 120
    
    # Load persistence handler
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    persistence = FilePersistence(save_file)
    
    # Load miner
    template_miner = TemplateMiner(persistence, config)

    return template_miner

def extract_log_templates(system: str, log_list: list, save_path: str, overwrite_miner = True) -> TemplateMiner:
    """
    Train a Drain3 miner over raw logs and persist its state to disk.

    Initializes a `TemplateMiner` (optionally resetting prior state), feeds each
    log line, and persists the trained miner. Prints extracted templates sorted
    by frequency and emits a profiler report.

    Args:
        system (str): System identifier used for config and persistence naming.
        log_list (list[str]): Raw log lines to process.
        save_path (str): Directory where the miner state file will be stored.
        overwrite_miner (bool): If True and a prior state exists, delete it to retrain.

    Returns:
        TemplateMiner: Trained miner with extracted log templates.
    """
    # Initialize the Drain miner
    drain_persistence_file = os.path.join(save_path, f"drain3-{system}-state.bin")
    if overwrite_miner and os.path.exists(drain_persistence_file):
        os.remove(drain_persistence_file)
        logging.info("Removed previous drain miner.")
        
    miner: TemplateMiner = init_drain(system, drain_persistence_file)

    # Feed logs to Drain
    for line in tqdm(log_list, desc="Add logs to Drain miner"):
        miner.add_log_message(line.rstrip())

    logging.info(f"Extracted {len(miner.drain.clusters)} templates.")

    # Save the trained miner to disk
    miner.save_state(f"{system} logs")
    
    sorted_clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=True)
    logging.info("Templates by frequency")
    for cluster in sorted_clusters:
        print(cluster)

    miner.profiler.report(0)
    return miner