"""
rca_eval_judge_local.py

Evaluate 'RCA Inference' outputs using a locally-hosted LLM-as-a-Judge (Ollama).

Purpose:
- Load RCA execution samples (chat history + final answers) from CSV
- Prompt a local judge LLM with structured templates
- Score tasks such as failure classification and alert grounding
- Write judgments to JSONL for later analysis

Requirements:
- Ollama installed and judge models pulled locally
- Config YAML (default: code/config/llm-judge-config.yaml)
- Samples CSV provided via config settings

Key Options:
- --datasets, --models, --judge-models, --tasks, --sample-size, --scores-file, --log-file

Notes:
- Runs fully locally; no external API calls
- Progress-aware logging via tqdm

Examples:
    python code/rca_eval_judge_local.py --config code/config/llm-judge-config.yaml
    python code/rca_eval_judge_local.py --datasets MicroSS --models llama3.3 --judge-models qwen3:32 --tasks failure-classification

"""

import os
import json
import argparse
import logging
import time
from typing_extensions import Annotated
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from pandas.api.types import is_scalar
from typing import TypedDict, List
from utils.logging import setup_logger
from utils.io import load_config, load_prompt_templates, write_to_file
from utils.llm import ExecutedToolCall, load_messages_from_string, load_model_ollama

@dataclass
class LLMJudgeResult:
    """
    Represents the result of an LLM-as-a-judge evaluation for a SINGLE sample.
    
    Attributes:
        id (str): Unique identifier of the sample.
        dataset (str): Name of the dataset to which the sample belongs.
        system (str): Name of the system to which the sample belongs.
        model (str): Name of the LLM model that generated the RCA result.
        setting (str): Experimental setting identifier that generated the RCA result.
        fault_id (str): ID of the specific fault scenario.
        modalities (str): Data modalities available to the agent that generated the RCA result.
        kg_rep (str): Knowledge graph representation type.
        alerts_rep (str): Alert representation type.
        task (str): The evaluation task performed (e.g., 'failure-classification').
        judge_model (str): Name of the LLM used as the judge.
        messages (str | List[BaseMessage]): The prompt/conversation sent to the judge model.
        final_response (str | dict | None): The output from the judge model.
    """
    # Metadata of sample
    id: str
    dataset: str
    system: str
    model: str
    setting: str
    fault_id: str
    modalities: str
    kg_rep: str
    alerts_rep: str
    # Judge output
    task: str
    judge_model: str
    messages: str
    final_response: str # should be a string or json object -- not a message.

class ReferencedAlerts(TypedDict):
    """
    Structure for 'alert-grounding' task output.
    
    Attributes:
        referenced_alerts: A list of lists, where each inner list contains 
                           identifiers for a single referenced alert.
    """
    # Referenced alerts by the agent
    referenced_alerts: List[List[str]]  # Each item in the list corresponds to a single referenced alert.

class Annotation(TypedDict):
    """
    Structure for a SINGLE annotation for the 'failure-classification' task.
    
    Attributes:
        RF_label: The root failure identifier/label.
        model_claim: The specific claim made by the model in the chat history.
        analysis: Justification or evidence that the RF applies.
    """
    RF_label: Annotated[str, ..., "The RF identifier"]
    model_claim: Annotated[str, ..., "The claim made by the model in the chat history"]
    analysis: Annotated[str, ..., "The justification or matching evidence thatt the RF applies."]

class AnnotatedFailures(TypedDict):
    """
    Structure for the 'failure-classification' task output.
    
    Attributes:
        annotations: List of individual annotations.
        primary_label: The primary root failure label identified.
        overall_severity: The overall severity of the failure.
        summary: Summary of annotations and rationale.
    """
    annotations: Annotated[List[Annotation], [], "The list of ALL annotations"]
    primary_label: Annotated[str, ..., "The primary RF"]
    overall_severity: Annotated[str, ..., "The overal severity"]
    summary: Annotated[str, ..., "Summary of annotations and rationale for the primary label"]


def is_none(x):
    """
    Check if a value effectively represents 'None' or 'NaN' in a pandas context.

    Args:
        x (Any): The value to check.

    Returns:
        bool: True if x is None, empty string, or NaN; False otherwise.
    """
    if not x:
        return True
    if is_scalar(x):
        return pd.isna(x)
    else:
        # For arrays/Series, treat any NaN present as not usable.
        return bool(pd.isna(x).any())
    
def safe_eval(x):
    """
    Safely evaluate a string representation of a Python literal using pd.eval.
    
    Returns None if the input is None or empty.

    Args:
        x (str | Any): The string to evaluate.

    Returns:
        Any: The evaluated Python object, or None.
    """
    if is_none(x) or x=='':
        return None
    return pd.eval(x)

def load_df_llm(file):
    """
    Load RCA LLM execution results from a CSV file into a pandas DataFrame.
    
    Handles safe evaluation of list/dict columns and deserialization of 
    ToolCall objects and LangChain message histories.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded samples with parsed columns.
    """
    logging.info(f"Loading selected samples from {file}")
    df_llm = pd.read_csv(
        file,
        index_col=0,
        dtype={
            'fault_id': 'str',
            'llm_values_count': 'int',
        },
        converters={
            'messages_types': safe_eval,
            'll_values_count': safe_eval,
            'num_tool_calls': safe_eval,
            'num_tool_calls_success': safe_eval,
            'num_tool_calls_error': safe_eval,
            'num_tool_calls_not_run': safe_eval,
            'path_hops_count': safe_eval,
            'path_failed_hops_count': safe_eval,
            'path_hops': safe_eval,
            'tool_calls': (lambda x: [ExecutedToolCall(**obj) for obj in json.loads(x)] if not is_none(x) else None),
            'failed_tool_calls': (lambda x: [ExecutedToolCall(**obj) for obj in json.loads(x)] if not is_none(x) else None),
        }
    )
    
    df_llm["messages"] = df_llm["messages_str"].apply(load_messages_from_string)
    
    return df_llm


def invoke_for_task(
    task: str, 
    prompt_templates, 
    messages: list[BaseMessage], 
    sample, 
    judge_model_name: str, 
    judge_model: BaseChatModel
) -> LLMJudgeResult:
    """
    Run a single evaluation task for a given sample using the local judge model.

    Constructs prompts from templates and the sample's chat history/ground truth,
    invokes the judge via Ollama, and packages the result.

    Args:
        task (str): Task ID (e.g., 'failure-classification' or 'alert-grounding').
        prompt_templates (dict): Dict of prompt templates keyed by task.
        messages (list[BaseMessage]): Message history from the RCA execution.
        sample (pd.Series): Sample row with ground truth fields.
        judge_model_name (str): Judge model name/ID.
        judge_model (BaseChatModel): Instantiated judge model.

    Returns:
        LLMJudgeResult: Result containing the judgment and metadata.
    """
    chat_history = ""
    for message in messages:
        chat_history += f"{message.pretty_repr()}\n\n"

    final_answer_message: BaseMessage = messages[-1]
    if not (final_answer_message.type == 'ai'):
        logging.warning(f"Sample {sample['id']}: Last message is not 'ai'.")

    all_data = {
        "chat_history": chat_history,
        "root_cause_hypotheses": final_answer_message.content,
        "root_cause_location": sample["fault_entity"],
        "root_cause_type": sample["fault_type"],
        "final_answer": sample["final_response"]
    }
    instructions = [
        SystemMessage(content=prompt_templates[task]['system'].format(**all_data)),
        HumanMessage(content=prompt_templates[task]['human'].format(**all_data))
    ]
    match task:
        case "failure-classification":
            # response = judge_model.with_structured_output(AnnotatedFailures).invoke(instructions)
            # messages = instructions
            # final_response = response
            response = judge_model.invoke(instructions)
            messages = instructions + [response]
            final_response = None
        case "alert-grounding":
            response = judge_model.with_structured_output(ReferencedAlerts).invoke(instructions)
            messages = instructions
            final_response = response

    result = LLMJudgeResult(
        id=sample['id'],
        dataset=sample['dataset'],
        system=sample['system'],
        model=sample['model'],
        setting=sample['setting'],
        fault_id=sample['fault_id'],
        modalities=sample['modalities'],
        kg_rep=sample['kg_rep'],
        alerts_rep=sample['alerts_rep'],
        task=task,
        judge_model=judge_model_name,
        messages=messages,
        final_response=final_response # Cannot be of type Message -- either string or json.
    )
    return result

def parse_args():
    """
    Parse command-line arguments for the local LLM-as-a-Judge script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RCA eval")
    parser.add_argument('--config', type=str, default="code/config/llm-judge-config.yaml", help="Path to configuration YAML.")
    parser.add_argument('--datasets', type=str, help="Name of the system(s) to evaluate, one of: 'MicroSS', 'OpenRCA-Market'")
    parser.add_argument('--models', type=str, help="Model results to evaluate.")
    parser.add_argument('--judge-models', type=str, help="Name of the model to act as the judge.")
    parser.add_argument('--tasks', type=str, help="Tasks to run.")
    parser.add_argument('--sample-size', type=str, help="Sample size PER dataset/model/setting")
    parser.add_argument('--scores-file', type=str, help="File name to write results to")
    parser.add_argument('--log-file', type=str)
    return parser.parse_args()
    
def main():
    """
    Main execution entry point for local LLM-as-a-Judge evaluation.
    
    - Loads configuration and arguments.
    - Sets up logging with tqdm-friendly handler.
    - Loads samples DataFrame.
    - Instantiates judge model via Ollama.
    - Iterates tasks/datasets/models and evaluates each sample.
    - Writes results periodically to JSONL.
    """

    start_time = time.time()
    args = parse_args()
    logger = logging.getLogger()
    processed = 0

    try:
        config = load_config(args.config)
        log_file = args.log_file or config['log-file']
        scores_file = args.scores_file or config['scores-file']
        logger = setup_logger(log_file)
        
        logger.info("Starting llm_judge")
        
        tasks = args.tasks.split(" ") if args.tasks else config["tasks-to-run"]
        datasets = args.datasets.split(" ") if args.datasets else config["datasets"]
        models = args.models.split(" ") if args.models else config["models"]
        judge_model_names = args.judge_models.split(" ") if args.judge_models else config['judge-models']

        logger.info(f"Executing: judge models={judge_model_names}; tasks={tasks}; datasets={datasets}; models={models}")
        
        prompts_dir = config["prompts-dir"]
        prompt_templates = load_prompt_templates(prompts_dir)['llm_judge']

        samples_file = os.path.join(config['data-directory'], config['samples-file'])
        df_judge_samples = load_df_llm(samples_file)
        samples_to_execute = df_judge_samples[
            df_judge_samples['dataset'].isin(datasets) &
            df_judge_samples['model'].isin(models)
        ]

        num_samples_to_execute = len(samples_to_execute) * len(judge_model_names) * len(tasks)
        
        logger.info(f"Loaded {len(df_judge_samples)} total samples from {samples_file}")
        logger.info(f"Executing samples: {num_samples_to_execute}")
        
        results: list[LLMJudgeResult] = []
        with tqdm(total=num_samples_to_execute, desc="LLM-Judge") as pbar:
            for judge_model_name in judge_model_names:
                logging.info(f"--------------------------------------------------------------------------")
                logging.info(f"Using {judge_model_name} as LLM Judge")
                judge_model = load_model_ollama(
                    judge_model_name,
                    num_ctx=config['num-ctx'],
                    max_new_tokens=config['max-new-tokens'],
                    reasoning=True
                )

                results_dir = config["results-directory"].format(judge_model=judge_model_name)
                os.makedirs(results_dir, exist_ok=True)
                results_file = os.path.join(results_dir, scores_file)

                for task in tasks:
                    for dataset in datasets:
                        for model in models:
                            samples = df_judge_samples.loc[
                                (df_judge_samples['dataset'] == dataset) &
                                (df_judge_samples['model'] == model)
                            ]
                            logging.info(f"----------------------------------")
                            logging.info(f"Parameters: task={task}, dataset={dataset}, model={model}")

                            for i, sample in samples.reset_index().iterrows():
                                processed += 1
                                pbar.update(1)
                                meter = pbar.format_meter(pbar.n, pbar.total, pbar.format_dict.get("elapsed", 0))
                                logging.info(f"{meter}")

                                result = invoke_for_task(task, prompt_templates, sample['messages'], sample, judge_model_name, judge_model)
                                results.append(result)

                                # Save results according to write-results-freq
                                if processed % config['write-results-freq'] == 0:
                                    logging.info(f"Writing {len(results)} to file {results_file}")
                                    for result in results:
                                        write_to_file(result, results_file, include_print=False)
                                    results = []

                # Save any remaining results for this judge model
                if results:
                    logging.info(f"Writing {len(results)} remaining results to file {results_file}")
                    for result in results:
                        write_to_file(result, results_file, include_print=False)
                    results = []
                    
    except Exception as e:
        logger.error(f"ERROR: Script crashed or existed early", exc_info=True)
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        logger.info("--------- LLM JUDGE COMPLETE ---------")
        logger.info(f"Processed samples: {processed}")
        logger.info(f"End time: {end_time}. Total time taken: {minutes} minutes {seconds} seconds ({minutes / 60} h)")

if __name__ == "__main__":
    main()
   