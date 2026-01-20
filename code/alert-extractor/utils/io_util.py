"""
I/O Utilities for Alert Extraction
"""
import json
import pickle
import yaml
import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load(file):
    """
    Load a Python object from a pickle file.

    Args:
        file (str | Path): Path to the pickle file.

    Returns:
        Any: The deserialized Python object.
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def load_config(config_path) -> dict:
    """
    Load configuration from a YAML file.

    Logs informative messages and gracefully handles file-not-found and YAML
    parsing errors, returning an empty dict on failure.

    Args:
        config_path (str | Path): Path to a YAML file containing dataset and
            processing configuration (e.g., processed-data paths, gt-file,
            run-table, normal-period).

    Returns:
        dict: Parsed configuration dictionary; empty dict on failure.
    """
    logging.info(f"Loading config file from {config_path}...")
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        logging.error(f"ERROR: config file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"ERROR: failed to parse YAML from {config_path}: {e}")
        return {}

def save(file, data):
    """
    Save a Python object to a pickle file using the highest protocol.

    Ensures the destination directory exists prior to writing.

    Args:
        file (str | Path): Output path for the pickle file.
        data (Any): Serializable Python object.
    """
    # Ensure the directorry exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file: str) -> dict:
    """
    Load and parse a JSON file.

    Args:
        file (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the file contents are not valid JSON.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, 'r') as f:
        data = json.load(f)
    logging.info(f'Loaded successfully from {file}!')
    return data

def save_json(file, data: dict):
    """
    Serialize and write a dictionary to a JSON file.

    Ensures the destination directory exists prior to writing.

    Args:
        file (str | Path): Output path for the JSON file.
        data (dict): Serializable dictionary.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as f:
        json.dump(data, f)
    logging.info(f'Saved successfully to {file}!')
    

def sanitize_file_dir_name(name: str) -> str:
    """
    Make a flow key safe for filesystem paths by replacing slashes.

    Args:
        name (str): Original flow key (e.g., "parent/service/operation").

    Returns:
        str: Sanitized name with '/' replaced by '_SL_'.
    """
    return name.replace('/', '_SL_')

def unsanitize_file_dir_name(name: str) -> str:
    """
    Restore original flow key by converting sanitized separators back to slashes.

    Args:
        name (str): Sanitized name containing '_SL_'.

    Returns:
        str: Original flow key with '/' separators.
    """
    return name.replace('_SL_', '/')

def save_fault_data(output_dir: str, fault_data: dict):
    """
    Save per-fault telemetry partitions by modality (log, trace, metric).

    Produces one parquet file per fault ID and modality under
    <output_dir>/<modality>/<fault_id>.parquet. Indices are reset to ensure
    clean, column-aligned storage.

    Args:
        output_dir (str): Root directory where data will be saved.
        fault_data (dict): Mapping of the form:
            {
                fault_id: {
                    "log": pd.DataFrame,
                    "trace": pd.DataFrame,
                    "metric": pd.DataFrame
                }
            }

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fault_id, modalities in tqdm(fault_data.items(), desc="Saving fault data"):
        fault_id = str(fault_id)

        # Save log, trace and metrics
        for modality in ['log', 'trace', 'metric']:
            if modality in modalities:
                path = output_dir / modality / f"{fault_id}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                modalities[modality].reset_index().to_parquet(path, index=False, engine="pyarrow")
            else:
                print(f"Skipping unrecognized modality: {modality} for fault {fault_id}")
    
def load_fault_data(input_dir: str, modalities_to_skip: list[str] = [], fault_ids_to_include = None) -> dict:
    """
    Load fault-partitioned telemetry into a nested dictionary structure.

    Reads parquet files saved by save_fault_data() and reconstructs a mapping
    {fault_id: {modality: pd.DataFrame}}. Optionally restricts to a subset of
    fault_ids_to_include.

    Args:
        input_dir (str): Root directory containing saved fault data.
        modalities_to_skip (list[str]): Modalities ('log', 'trace', 'metric') to skip.
        fault_ids_to_include (Iterable[int] | None): If provided, only these fault
            IDs are loaded; otherwise all available files are read.

    Returns:
        dict: Nested mapping Dict[int][str] -> pd.DataFrame.
    """
    input_dir = Path(input_dir)
    fault_data = {}

    # Load log and trace
    for modality in ['log', 'trace', 'metric']:
        if modality in modalities_to_skip:
            continue
        modality_dir = input_dir / modality
        if modality_dir.exists():
            if not fault_ids_to_include:
                for file in tqdm(list(modality_dir.glob("*.parquet")), desc=f"Loading {modality} files"):
                    fault_id = int(file.stem)
                    
                    data_for_fault = pd.read_parquet(file, engine="pyarrow")
                        
                    fault_data.setdefault(fault_id, {})[modality] = data_for_fault
            else:
                for fault_id in tqdm(fault_ids_to_include, desc=f"Loading {modality} files"):
                    file = os.path.join(modality_dir, f"{str(fault_id)}.parquet")
                    
                    data_for_fault = pd.read_parquet(file, engine="pyarrow")
                        
                    fault_data.setdefault(int(fault_id), {})[modality] = data_for_fault
        else:
            print(f"{modality_dir} does not exist.")
    
    return fault_data

def save_normal_data(output_dir: str, normal_data: dict):
    """
    Save normal (non-fault) telemetry per modality (log, trace, metric).

    Writes one parquet file per modality to <output_dir>/<modality>.parquet.
    Indices are reset to ensure clean, column-aligned storage.

    Args:
        output_dir (str): Directory to save the data to.
        normal_data (dict): Dictionary with structure:
            {
                "log": pd.DataFrame,
                "trace": pd.DataFrame,
                "metric": pd.DataFrame
            }

    Returns:
        None
    """
    output_dir = Path(output_dir)

    # Save log, trace and metrics
    for modality in ['log', 'trace', 'metric']:
        if modality in normal_data:
            path = output_dir / f"{modality}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normal_data[modality].reset_index().to_parquet(path, index=False, engine="pyarrow")
        else:
            print(f"Skipping unrecognized modality: {modality}")

def load_normal_data(input_dir: str, modalities_to_skip: list[str] = []) -> dict:
    """
    Load normal (non-fault) telemetry per modality.

    Reads parquet files saved by save_normal_data() and returns a mapping
    {modality: pd.DataFrame} for available modalities.

    Args:
        input_dir (str): Directory where normal data is stored.
        modalities_to_skip (list[str]): Modalities to skip (e.g., ['log', 'trace', 'metric']).

    Returns:
        dict: Mapping Dict[str] -> pd.DataFrame for loaded modalities.
    """
    input_dir = Path(input_dir)
    data = {}

    # Load log and trace
    for modality in ['log', 'trace', 'metric']:
        if modality in modalities_to_skip:
            continue
        file = input_dir / f"{modality}.parquet"
        if file.exists():
            data[modality] = pd.read_parquet(file)

    return data

def save_trace_detectors(output_dir: str, detectors: dict):
    """
    Persist trained trace anomaly detectors to disk.

    Saves each flow's detectors to a dedicated subdirectory under output_dir.
    The subdirectory name is derived from the flow key via sanitize_file_dir_name().

    Args:
        output_dir (str): Root directory for detector files.
        detectors (dict): Nested mapping of the form
            {flow_key: {detector_name: IsolationForest}}.

    Returns:
        None
    """
    output_dir = Path(output_dir)

    for flow_key, flow_detectors in detectors.items():
        flow_dir = output_dir / sanitize_file_dir_name(flow_key)
        flow_dir.mkdir(parents=True, exist_ok=True)
        for name, detector in flow_detectors.items():
            path = flow_dir / f"{name}.pkl"
            save(path, detector)
            
def load_trace_detectors(input_dir: str) -> dict:
    """
    Load trained trace anomaly detectors from disk.

    Expects a directory structure where each flow has its own folder containing
    one or more `*.pkl` files. Flow folder names are unsanitized back to their
    original keys via unsanitize_file_dir_name().

    Args:
        input_dir (str): Root directory where trace detectors are stored.

    Returns:
        dict: Nested mapping {flow_key: {detector_name: IsolationForest}}.
    """
    
    input_dir = Path(input_dir)
    all_detectors = {}

    for flow_dir in input_dir.iterdir():
        if flow_dir.is_dir():
            flow_key = unsanitize_file_dir_name(flow_dir.name)
            detectors = {}
            for file in flow_dir.glob("*.pkl"):
                detector_name = file.stem
                detectors[detector_name] = load(file)
            all_detectors[flow_key] = detectors

    return all_detectors