"""
rca_eval_judge_openai.py

Evaluate 'RCA Inference' outputs using the OpenAI Batch API for LLM-as-a-Judge.

Purpose:
- Load RCA execution samples (chat history + final answers) from CSV
- Format per-sample OpenAI Chat Completions requests into JSONL batches
- Upload batch files and create Batch jobs
- Check batch status and retrieve outputs/errors
- Persist batch metadata and results for later analysis

Requirements:
- OpenAI API access and environment configured (e.g., OPENAI_API_KEY)
- Config YAML with endpoint and model settings (e.g., /v1/chat/completions)
- Samples CSV provided via config settings

Key Options:
- --judge-model, --task, --prep-batch-files, --kickoff-batches, --check-status,
    --batch-files-glob, --batch-ids-file, --batch-size, --num-batches, --scores-file, --log-file

Notes:
- Uses batched, asynchronous processing; results may require later retrieval
- Batch input/output files are written under the configured data directory
- Supports reasoning parameters such as 'reasoning_effort' and 'max_completion_tokens'

Examples:
    # Prepare batch files
    python code/rca_eval_judge_openai.py --config code/config/llm-judge-config.yaml --judge-model gpt-5 --task failure-classification --prep-batch-files --batch-size 1000 --num-batches 4
    # Kick off batches
    python code/rca_eval_judge_openai.py --config code/config/llm-judge-config.yaml --kickoff-batches
    # Check status and retrieve results
    python code/rca_eval_judge_openai.py --config code/config/llm-judge-config.yaml --check-status

"""

import os
import json
import argparse
import logging
import time
import glob
from openai import OpenAI
from openai.types import FileObject, Batch
from langchain_core.messages import BaseMessage
from typing import List

from utils.io import load_config, load_prompt_templates
from rca_eval_judge_local import load_df_llm
from utils.logging import setup_logger


def format_samples_for_openai(
    llm_judge_data,
    prompt_templates,
    task="failure-classification",
    method="POST",
    url="/v1/chat/completions",
    model="gpt-5-mini",
    reasoning_effort: str | None = "high",
    max_completion_tokens: int | None = 30000
):
    """
    Format RCA samples (generated during RCA Inference) into OpenAI Chat Completions batch requests.

    Constructs per-sample request objects with `messages` built from prompt templates
    and each sample's chat history and final answer.

    Args:
        llm_judge_data (pd.DataFrame): Samples for judge to evaluate (rows include `messages` and metadata).
        prompt_templates (dict): Prompt templates keyed by task and role (system or human).
        task (str): Task name (e.g., "failure-classification" or "alert-grounding").
        method (str): HTTP method, typically "POST".
        url (str): Endpoint path (e.g., "/v1/chat/completions").
        model (str): OpenAI model ID for judging (e.g., "gpt-5-mini").
        reasoning_effort (str | None): Reasoning configuration for compatible models.
        max_completion_tokens (int | None): Cap for completion tokens.

    Returns:
        List[dict]: Request dictionaries suitable for JSONL batch files.
    """
    requests = []
    for index, sample in llm_judge_data.iterrows():
        messages = sample['messages']
        chat_history = ""
        for message in messages:
            chat_history += f"{message.pretty_repr()}\n\n"

        final_answer_message: BaseMessage = messages[-1]
        
        all_data = {
            "chat_history": chat_history,
            "root_cause_hypotheses": final_answer_message.content,
            "root_cause_location": sample["fault_entity"],
            "root_cause_type": sample["fault_type"],
            "final_answer": sample["final_response"]
        }
        messages = [
            {
                "role": "system",
                "content": prompt_templates[task]['system'].format(**all_data)
            },
            {
                "role": "user",
                "content": prompt_templates[task]['human'].format(**all_data)
            }
        ]
        request = {
            "custom_id": str(index),
            "method": method,
            "url": url,
            "body": {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_completion_tokens,
                "messages": messages,
                
            }
        }
        requests.append(request)
    return requests

def write_batches_jsonl(
    formatted_samples: List[dict],
    output_dir: str,
    batch_size: int,
    num_batches: int,
    pattern: str = "openai_batch_{batch_index}_{timestamp}",
) -> List[str]:
    """
    Write formatted OpenAI batch requests into multiple JSONL files.

    Splits the list of request dictionaries into several files under `output_dir`
    according to `batch_size` and `num_batches`.

    Args:
        formatted_samples (List[dict]): Request dicts, one per sample.
        output_dir (str): Directory to write files into (created if missing).
        batch_size (int): Max number of items per file.
        num_batches (int): Number of files to create (last one may be partial).
        pattern (str): Filename pattern including `{batch_index}` and `{timestamp}`.

    Returns:
        List[str]: File paths created, in order.
    """
    os.makedirs(output_dir, exist_ok=True)

    total = len(formatted_samples)
    file_paths: List[str] = []

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        if start >= total:
            logging.warning(
                f"Requested {num_batches} batches, but only {i} could be created from {total} samples."
            )
            break
        batch_items = formatted_samples[start:end]
        timestamp = time.strftime("%Y%m%d-%H%M")
        batch_index = f"{i+1:03d}"
        file_name = f"{pattern.format(batch_index=batch_index, timestamp=timestamp)}.jsonl"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in batch_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Wrote batch {i+1} with {len(batch_items)} items to {file_path}")
        file_paths.append(file_path)

    leftover = total - min(total, batch_size * num_batches)
    if leftover > 0:
        logging.info(
            f"{leftover} samples were not included because num_batches * batch_size < total samples."
        )

    return file_paths

def prepare_batch_files(config, df_judge_samples, batch_size, num_batches, judge_model, task) -> list[str]:
    """
    Prepare JSONL batch files from DataFrame samples for OpenAI Batch.

    Splits formatted requests into multiple files according to `batch_size`
    and `num_batches`, writing them under the configured data directory.

    Args:
        config (dict): Configuration loaded from YAML.
        df_judge_samples (pd.DataFrame): Samples DataFrame for judging.
        batch_size (int): Max number of items per JSONL file.
        num_batches (int): Number of files to create.
        judge_model (str): Target OpenAI model ID for judging.
        task (str): Task name (e.g., "failure-classification").

    Returns:
        List[str]: Paths to the created batch JSONL files.
    """
    if (not batch_size) or (not num_batches):
        logging.error("No batch size or num batches provided.")
        raise Exception

    prompts_dir = config["prompts-dir"]
    prompt_templates = load_prompt_templates(prompts_dir)['llm_judge']
    endpoint = config['openai-endpoint']
    max_completion_tokens = config['max-completion-tokens']

    formatted_samples = format_samples_for_openai(
        df_judge_samples,
        prompt_templates,
        task=task,
        method="POST",
        url=endpoint,
        model=judge_model,
        max_completion_tokens=max_completion_tokens
    )
    output_dir = config['data-directory']
    batch_file_paths = write_batches_jsonl(
        formatted_samples=formatted_samples,
        output_dir=output_dir,
        batch_size=batch_size,
        num_batches=num_batches,
    )
    processed = min(len(formatted_samples), batch_size * num_batches)
    logging.info(
        f"Prepared {len(batch_file_paths)} batch file(s) with up to {batch_size} items each in '{output_dir}'."
    )
    return batch_file_paths
    
def find_batch_files(
    config,
    pattern: str = "openai_batch_*.jsonl"
)-> list[str]:
    """
    Find batch JSONL files to upload.

    Args:
        config (dict): Configuration with `data-directory` fallback.
        pattern (str): Glob pattern used to find batch files.

    Returns:
        List[str]: Sorted list of matching batch file paths.
    """
    paths = glob.glob(pattern)
    if not paths:
        base = config['data-directory']
        paths = glob.glob(os.path.join(base, pattern))

    paths = sorted(paths)
    logging.info(f"Discovered {len(paths)} batch file(s) to upload: {paths}")

    return paths

def upload_batches(batch_data_files: list[str]) -> list[FileObject]:
    """
    Upload local batch data files to OpenAI as `FileObject`s.

    Args:
        batch_data_files: Paths to JSONL batch files previously prepared.

    Returns:
        list[FileObject]: Uploaded file objects to be used for Batch creation.
    """
    client = OpenAI()
    def upload_batch_file(batch_data_file) -> FileObject:
        batch_input_file = client.files.create(
            file=open(batch_data_file, "rb"),
            purpose="batch"
        )
        logging.info(f"Batch input file: {batch_input_file}")
        return batch_input_file
    
    batch_input_files: list[FileObject] = []
    for batch_data_file in batch_data_files:
        batch_input_file = upload_batch_file(batch_data_file)
        batch_input_files.append(batch_input_file)
    
    logging.info(f"{len(batch_input_files)} batch input files uploaded.")
    return batch_input_files
        
def create_batches(
    config,
    batch_input_files: list[FileObject],
    completion_window: str = "24h",
    description: str | None = None,
) -> list[Batch]:
    """
    Create OpenAI Batch jobs from uploaded input files.

    Args:
        config (dict): Configuration dict containing `openai-endpoint`.
        batch_input_files (List[FileObject]): Uploaded input file objects.
        completion_window (str): Requested batch completion window (e.g., "24h").
        description (str | None): Optional description stored as batch metadata.

    Returns:
        List[Batch]: Created Batch objects (may be empty if errors occur).
    """
    client = OpenAI()
    endpoint = config['openai-endpoint']
    
    def create_batch(batch_input_file: FileObject):
        batch_input_file_id = batch_input_file.id
        
        try:
            batch: Batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata={"description": description} if description else None,
            )
            logging.info(f"Batch {batch.id} created from {batch_input_file_id}. Status = {batch.status}")
            return batch
        except Exception as e:
            logging.error(f"Error while creating batch.", exc_info=True)
            return None
    
    batches: list[Batch] = []
    for batch_input_file in batch_input_files:
        batch = create_batch(batch_input_file)
        if batch:
            batches.append(batch)
    
    return batches
        
def write_batch_ids(
    config,
    batches: list[Batch],
    batch_file_paths: list[str],
    out_file_pattern: str = "openai_batch_ids_{timestamp}.jsonl"
) -> str:
    """
    Write created batch IDs and metadata to a JSONL file in the data directory.

    Args:
        config (dict): Configuration with `data-directory` where file will be written.
        batches (List[Batch]): Batch objects recently created.
        batch_file_paths (List[str]): Local paths of input files corresponding to the batches.
        out_file_pattern (str): Filename pattern for the output JSONL.

    Returns:
        str: Path to the written JSONL file.
    """
    data_dir = config['data-directory']
    os.makedirs(data_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(data_dir, out_file_pattern.format(timestamp=timestamp))

    # Ensure we can map batches back to their originating local file paths via order
    count = min(len(batches), len(batch_file_paths))
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(count):
            batch = batches[i]
            batch_dict = batch.to_dict()
            batch_dict["logged_at"] = timestamp
            batch_dict["batch_file_path"] = batch_file_paths[i]

            f.write(json.dumps(batch_dict, ensure_ascii=False) + "\n")

    logging.info(f"Wrote {count} batch id record(s) to {out_path}")
    return out_path

def get_latest_batch_ids_file(
    config,
    pattern: str = "openai_batch_ids_*.jsonl"
)-> str | None:
    """
    Find the latest batch IDs file in the configured data directory.

    Args:
        config: Configuration dict containing `data-directory`.
        pattern: Glob pattern for batch IDs files.

    Returns:
        str | None: Path to the newest batch IDs file, or None if none found.
    """
    base = config['data-directory']
    candidates = glob.glob(pattern)
    if not candidates:
        candidates = glob.glob(os.path.join(base, pattern))
    
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]

def read_batch_ids_file(path: str) -> list[str]:
    """
    Read a JSONL file of batch metadata and extract batch IDs.

    Args:
        path: Path to the JSONL file containing batch records.

    Returns:
        list[str]: List of batch IDs contained in the file.
    """
    batch_ids: list[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                batch_id = obj.get("batch_id") or obj.get("id")
                if batch_id:
                    batch_ids.append(batch_id)
            except Exception:
                logging.warning(f"Could not parse line in batch IDs file: {line[:120]}...")
                continue

    logging.info(f"Loaded {len(batch_ids)} batch id(s) from {path}")
    return batch_ids

def update_batch_ids_file(batch_ids_file: str, updated_batches: list[Batch]) -> str:
    """
    Update batch records in the JSONL file with newer Batch objects.

    Merges fields and preserves existing metadata; writes to a temp file and
    atomically replaces the original file.

    Args:
        batch_ids_file (str): Path to the batch IDs JSONL file to update.
        updated_batches (List[Batch]): Latest Batch objects to merge into the file.

    Returns:
        str: The updated batch IDs file path.
    """
    if not updated_batches:
        return batch_ids_file

    # Read existing entries
    existing: list[dict] = []
    index_by_id: dict[str, int] = {}
    try:
        with open(batch_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                existing.append(obj)
                bid = obj.get("id") or obj.get("batch_id")
                if bid is not None:
                    index_by_id[str(bid)] = len(existing) - 1
    except FileNotFoundError:
        # No existing file; we'll create a new one
        pass

    now_ts = time.strftime("%Y%m%d-%H%M%S")
    for batch in updated_batches:
        batch_dict = batch.to_dict()

        bid = batch_dict["id"]
        if not bid:
            continue

        if bid in index_by_id:
            idx = index_by_id[bid]
            prev = existing[idx]
            existing[idx] = {**prev, **batch_dict, "logged_at": now_ts}
        else:
            existing.append({**batch_dict, "logged_at": now_ts})
            index_by_id[bid] = len(existing) - 1

    tmp_path = batch_ids_file + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for obj in existing:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, batch_ids_file)
    logging.info(f"Batch IDs file updated with {len(updated_batches)} batch record(s): {batch_ids_file}")
    return batch_ids_file

def check_status_and_retrieve(
    config,
    batch_ids_file: str | None = None,
    batch_ids: list[str] | None = None,
    output_file_pattern: str = "openai_batch_output_{batch_id}.jsonl",
    error_file_pattern: str = "openai_batch_errors_{batch_id}.jsonl",
    update_ids_file: bool = True,
):
    """
    Check status for given batch IDs and retrieve outputs/errors when ready.

    If `batch_ids_file` is provided, loads IDs from file; otherwise uses `batch_ids`.
    Saves outputs and errors to files under the configured data directory.

    Args:
        config (dict): Configuration with `data-directory` and `openai-endpoint`.
        batch_ids_file (str | None): Optional path to JSONL file containing batch IDs.
        batch_ids (List[str] | None): Optional explicit list of batch IDs to process.
        output_file_pattern (str): Filename template for outputs.
        error_file_pattern (str): Filename template for errors.
        update_ids_file (bool): Whether to update the batch IDs file with latest batch states.

    Returns:
        None
    """
    client = OpenAI()

    def write_bytes_to_file(path: str, data) -> None:
        # 'data' may be bytes, a stream, or an object with .content/.read.
        if hasattr(data, 'read'):
            content = data.read()
        elif hasattr(data, 'content'):
            content = data.content
        else:
            content = data
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(content)

    def retrieve_batch_results(batch: Batch):
        output_file_id: str = batch.output_file_id
        error_file_id: str = batch.error_file_id

        out_dir = config['data-directory']
        batch_id = batch.id
        output_path = os.path.join(out_dir, output_file_pattern.format(batch_id=batch_id))
        error_path = os.path.join(out_dir, error_file_pattern.format(batch_id=batch_id))

        if output_file_id:
            try:
                output_response = client.files.content(output_file_id)
                write_bytes_to_file(output_path, output_response)
                logging.info(f"Saved batch output to {output_path}")
            except Exception:
                logging.exception(f"Failed to download output for batch {batch_id}")
        else:
            logging.info(f"No output_file_id for batch {batch_id}")

        if error_file_id:
            try:
                error_response = client.files.content(error_file_id)
                write_bytes_to_file(error_path, error_response)
                logging.info(f"Saved batch errors to {error_path}")
            except Exception:
                logging.exception(f"Failed to download errors for batch {batch_id}")
        else:
            logging.info(f"No error_file_id for batch {batch_id}")
    
    if batch_ids_file:
        batch_ids = read_batch_ids_file(batch_ids_file)
    if not batch_ids:
        logging.error(f"No batch IDs found in file: {batch_ids_file}")
        return
    
    retrieved_batches: list[Batch] = []
    for batch_id in batch_ids:
        try:
            batch: Batch = client.batches.retrieve(batch_id)
            
            logging.info(f"Batch {batch.id}: {batch.status}")
            
            if batch.status == 'completed':
                logging.info(f"Completed at: {getattr(batch, 'completed_at', None)}")
            elif batch.status == 'cancelled':
                logging.info(f"Cancelled at: {getattr(batch, 'cancelled_at', None)}")
            
            if batch.status in ['completed', 'cancelled', 'expired', 'failed']:
                retrieve_batch_results(batch)
            else:
                logging.info("Not completed yet; skipping retrieval.")
            retrieved_batches.append(batch)
        except Exception:
            logging.exception(f"Failed to check or retrieve for batch {batch_id}")
    # Update batch IDs file with latest batch objects
    if update_ids_file and batch_ids_file:
        update_batch_ids_file(batch_ids_file, retrieved_batches)
            

def parse_args():
    """
    Parse command-line arguments for OpenAI Batch evaluation workflow.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Prep Data and Submit to OpenAI Batch API")
    parser.add_argument('--config', type=str, default="code/config/llm-judge-config.yaml", help="Path to configuration YAML.")
    parser.add_argument('--judge-model', type=str, help="Name of the model to act as the judge.")
    parser.add_argument('--task', type=str, default="failure-classification", help="Task to run.")
    parser.add_argument('--prep-batch-files', action='store_true', help="To prep batch file")
    parser.add_argument('--kickoff-batches', action='store_true', help="To upload and kickoff batches")
    parser.add_argument('--check-status', action='store_true', help="To check status. If complete, retrieve results.")
    parser.add_argument('--batch-files-glob', type=str, help="Glob pattern to find batch JSONL files for upload.")
    parser.add_argument('--batch-ids-file', type=str, help="Path to JSONL file with batch IDs (defaults to latest in data dir)")
    parser.add_argument('--batch-size', type=int, default=3, help="Number of samples to include in the batch.")
    parser.add_argument('--num-batches', type=int, default=1, help="Number of batches to prep and/or run")
    parser.add_argument('--scores-file', type=str, help="File name to write results to")
    parser.add_argument('--log-file', type=str)
    return parser.parse_args()
    
def main():
    """
    Main entry point: prepare/upload batches, or check status/retrieve outputs.

    Behavior depends on flags:
    - `--prep-batch-files`: format samples and write JSONL batches
    - `--kickoff-batches`: upload batch files and create Batch jobs
    - `--check-status`: check state of batches and download outputs/errors
    """
    start_time = time.time()
    args = parse_args()
    logger = logging.getLogger()
    processed = 0

    try:
        config = load_config(args.config)
        log_file = args.log_file or config['log-file']
        logger = setup_logger(log_file)

        to_prep_batch_data_files = args.prep_batch_files
        to_kickoff_batches = args.kickoff_batches
        to_check_status = args.check_status

        batch_file_paths = None

        if to_prep_batch_data_files:
            samples_file = os.path.join(config['data-directory'], config['samples-file'])
            df_judge_samples = load_df_llm(samples_file)
            
            # Randomly shuffle the samples! 
            df_judge_samples = df_judge_samples.sample(frac=1, random_state=42)
            logging.info(f"Shuffled judge samples with seed {42} (index preserved)")
            
            judge_model = args.judge_model or config['openai-judge-model']
            task = args.task
            
            batch_file_paths = prepare_batch_files(
                config,
                df_judge_samples,
                args.batch_size,
                args.num_batches,
                judge_model,
                task
            )
        
        if to_kickoff_batches:
            # Use prepared paths if present; otherwise discover via glob or default pattern.
            if not batch_file_paths:
                batch_file_paths = find_batch_files(config, args.batch_files_glob)
            if not batch_file_paths:
                logging.error("No batch file paths to upload. Prepare them first or provide --batch-files-glob.")
                return

            batch_input_files: list[FileObject] = upload_batches(batch_file_paths)
            batches: list[Batch] = create_batches(config, batch_input_files)
            
            # Write batch ids to file
            batch_ids_file = write_batch_ids(config, batches, batch_file_paths)
            logging.info(f"Batch IDs written to: {batch_ids_file}")
            
        if to_check_status:
            # Determine batch IDs file
            batch_ids_file = args.batch_ids_file or get_latest_batch_ids_file(config)
            if not batch_ids_file:
                logging.error("No batch IDs file found. Provide --batch-ids-file or run kickoff first.")
                return

            # Load and check statuses
            batch_ids = read_batch_ids_file(batch_ids_file)
            if not batch_ids:
                logging.error(f"No batch IDs found in file: {batch_ids_file}")
                return

            check_status_and_retrieve(
                config,
                batch_ids_file=batch_ids_file
            )

    except Exception as e:
        logger.error(f"ERROR: Script crashed or existed early", exc_info=True)
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        logger.info("--------- LLM JUDGE OPENAI COMPLETE ---------")
        logger.info(f"End time: {end_time}. Total time taken: {minutes} minutes {seconds} seconds ({minutes / 60} h)")

if __name__ == "__main__":
    main()
   