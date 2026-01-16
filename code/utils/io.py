import os
import json
import yaml
from langchain_core.load import dumps
from dataclasses import asdict
from fault_scenarios import LogAlert, MetricAlert, TraceAlert

## --------------------------- Load from file ---------------------------

def load_config(config_path) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary, or an empty dict on error.
    """
    print(f"Loading config file from {config_path}...")
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        print(f"ERROR: config file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR: failed to parse YAML from {config_path}: {e}")
        return {}

def load_json(path) -> json:
    """
    Load data from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: The JSON content loaded from the file.
    """
    print(f"Loading file from {path}...")
    with open(path, 'r') as f:
        file = json.load(f)
    return file

def load_prompt_templates(prompts_dir: str, setting_name: str | None=None) -> dict:
    """
    Retrieves the rca and structured output prompt templates from the prompts directory.
    If other prompt template files are present in the prompts directory (e.g., plan-execute, llm-judge), then those are loaded too.
    If the setting name is provided, then only the prompt templates for that setting are returned.
    
    Args:
        prompts_dir (str): The directory containing the prompt templates.
        setting_name (str | None): The name of the setting to retrieve templates for.

    Returns:
        dict: A dict containing the rca and structured output prompt templates, formatted as: 
        {
            "rca": {
                "<setting_name>": {
                    "system": "...",
                    "human": "..."
                }
            },
            "structured_output": {
                "system": "...",
                "human": "..."
            },
            "plan_execute": {
                "planner": "...",
                "replanner": "..."
            },
            "llm_judge": {
                "system": "...",
                "human": "..."
            }
        }
        If setting_name is provided, then the "setting_name" intermediate key is not provided.
    """
    # Define file paths
    rca_file = os.path.join(prompts_dir, "rca-prompts.yaml")
    structured_output_file = os.path.join(prompts_dir, "structured-output-prompts.yaml")
    plan_execute_file = os.path.join(prompts_dir, "plan-execute-prompts.yaml")
    llm_judge_file = os.path.join(prompts_dir, "llm-judge-prompts.yaml")

    # Check if files exist
    if not os.path.isfile(rca_file):
        raise FileNotFoundError(f"No rca prompt template found: {rca_file}")
    if not os.path.isfile(structured_output_file):
        raise FileNotFoundError(f"No structured output prompt template found: {structured_output_file}")
    
    plan_execute_file_exists = True
    if not os.path.isfile(plan_execute_file):
        plan_execute_file_exists = False
        print(f"No prompt template file for the plan-execute agent is found: {plan_execute_file}")
    
    llm_judge_file_exists = True
    if not os.path.isfile(llm_judge_file):
        llm_judge_file_exists = False
        print(f"No prompt template file for the llm-judge is found: {llm_judge_file}")
    
    # Load files
    with open(rca_file, 'r') as f:
        rca_prompt_templates = yaml.safe_load(f)
    
    with open(structured_output_file, 'r') as f:
        so_prompt_templates = yaml.safe_load(f)
        
    if setting_name:
        return_dict = {"rca": rca_prompt_templates[setting_name], "structured_output": so_prompt_templates}
    else:
        return_dict = {"rca": rca_prompt_templates, "structured_output": so_prompt_templates}
    
    if plan_execute_file_exists:
        with open(plan_execute_file, 'r') as f:
            plan_execute_templates = yaml.safe_load(f)
        return_dict["plan_execute"] = plan_execute_templates
    
    if llm_judge_file_exists:
        with open(llm_judge_file, 'r') as f:
            llm_judge_templates = yaml.safe_load(f)
        return_dict["llm_judge"] = llm_judge_templates
        
    return return_dict


def load_failure_patterns(path: str) -> str:
    """
    Loads failure patterns from a specified file.

    Args:
        path (str): The path to the file containing failure patterns.

    Returns:
        str: The contents of the failure patterns file.
    """
    try:
        with open(path, 'r') as f:
            failure_patterns = f.read()
        return failure_patterns
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    
def load_fault_data(log_file: str, traces_file: str, metrics_file: str, modality_to_exclude: str = None) -> dict:
    """
    Loads fault data from log, trace, and metric files, optionally excluding one modality.
    
    Args:
        log_file (str): Path to the JSON file containing log alerts.
        traces_file (str): Path to the JSON file containing trace alerts.
        metrics_file (str): Path to the JSON file containing metric alerts.
        modality_to_exclude (str, optional): Modality to exclude from loading ('logs', 'traces', or 'metrics'). Defaults to None.
    
    Returns:
        dict: A dictionary mapping fault IDs to their corresponding alerts, structured as:
            {
                fault_id: {
                    "log_alerts": List[LogAlert],
                    "trace_alerts": List[TraceAlert],
                    "metric_alerts": List[MetricAlert]
                },
                ...
            If a modality is excluded, its alert list will be *empty*.
    """
    modalities = {
        "logs": log_file,
        "traces": traces_file,
        "metrics": metrics_file
    }
    data = {
        modality: {} if modality_to_exclude == modality 
        else load_json(file) 
        for modality, file in modalities.items()
    }
    logs, traces, metrics = data["logs"], data["traces"], data["metrics"]
    
    fault_data = {}
    
    # Collect all unique fault ids from the three sources
    fault_ids = set(logs.keys()) | set(traces.keys()) | set(metrics.keys())
    
    for fault_id in fault_ids:
        fault_data[fault_id] = {
            "log_alerts": list(map(lambda event: LogAlert(event[0], event[1], event[2], event[3], event[4]), logs.get(fault_id, []))),
            "trace_alerts": list(map(lambda event: TraceAlert(event[0], event[1], event[2], event[3], event[4]),traces.get(fault_id, []))),
            "metric_alerts": list(map(lambda event: MetricAlert(event[0], event[1], event[2], event[3], event[4]), metrics.get(fault_id, [])))
        }
    
    return fault_data


def retrieve_files_from_directory(dir_path):
    """
    Retrieves all files from a specified directory.

    Args:
        dir_path (str): The path to the directory.

    Returns:
        list: A list of file paths within the directory. Empty if none found or invalid path.
    """
    try:
        if not os.path.isdir(dir_path):
            raise ValueError(f"The provided path '{dir_path}' is not a valid directory.")

        files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []

## --------------------------- Save to file ---------------------------
def write_to_file(data, file_path="./samples.txt", include_print=True):
    """
    Append data to a file in JSON format.
    Handles lists by recursively calling itself for each item.
    Handles single objects by converting to dict (if needed), serializing messages (if present),
    and appending as a JSON line.

    Args:
        data: The data content to write (dict, list, or dataclass).
        file_path (str): Destination file path. Defaults to "./samples.txt".
        include_print (bool): Whether to print a confirmation message. Defaults to True.
    """
    if include_print:
        print(f"Writing responses to {file_path}")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a', encoding='utf-8') as f_converted:
        if isinstance(data, list):
            for item in data:
                write_to_file(item, file_path, include_print)
        else:
            # Handle single object case
            if not isinstance(data, dict):
                data = asdict(data)  # Convert to dictionary
            if data.get("messages", None):
                data["messages"] = dumps(data["messages"])   
            json_str = json.dumps(data)
        f_converted.write(json_str)
        f_converted.write('\n')
